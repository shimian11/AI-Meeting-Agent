import re
from enum import Enum
from typing import Any

import httpx
import openai
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.errors import AgentError, ErrorCode


class Tier(str, Enum):
    """模型档位"""
    FAST="fast"
    STRONG="strong"

class Purpose(str, Enum):
    """调用目的"""
    CHAT = "chat"           #聊天
    EVAL = "eval"           #评估（意图识别）
    GENERATE = "generate"   #出题
    SCORE = "score"         #简历评分

# 温度分档：对话 0.7 / 评分与意图 0.2 / 出题 0.5
TEMPERATURE: dict[Purpose, float] = {
    Purpose.CHAT: 0.7,
    Purpose.EVAL: 0.2,
    Purpose.GENERATE: 0.5,
    Purpose.SCORE: 0.2,
}

# 允许先打 fast 的聊天和评估；出题/评分直接 strong
FAST_ALLOWED: frozenset[Purpose] = frozenset({Purpose.CHAT, Purpose.EVAL})

#进程内降级计数
_fallback_counts: dict[str, int] = {}

# ```json ... ``` / ``` ... ````，将模型返回的json型响应只剥最外层
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.S)

def strip_code_fence(text: str) -> str:
    """
    剥掉 markdown 代码围栏：```json\\n{...}\\n``` -> {...}
    这是 with_structured_output 失败后的最后一道手工兜底，
    也对应 Java 侧 InterviewResponseParser 的同类逻辑（1C）。
    """
    if not text:
        return ""
    matched = _FENCE_RE.match(text)
    return matched.group(1) if matched else text.strip()

def fallback_counts() -> dict[str, int]:
    #返回降级计数器
    return dict(_fallback_counts)

def _record_fallback(reason: str) -> None:
    """记一次降级：跨 provider 后一次降级会同时消耗两家配额，必须可归因。"""
    _fallback_counts[reason] = _fallback_counts.get(reason, 0) + 1
    logger.warning(f"llm_fallback reason={reason} total={_fallback_counts[reason]}")

def _provider_of(tier: Tier) -> tuple[str, str, str, str]:
    """
    按档位返回 (provider 名, base_url, api_key, model)。
    关键行：四元组成组返回
    """
    s = get_settings()
    if tier is Tier.FAST:
        if not s.siliconflow_api_key:
            raise AgentError(
                ErrorCode.LLM_ERROR,
                "BANMA_SILICONFLOW_API_KEY 未配置（fast 档与 embedding 需要）",
            )
        return "siliconflow", s.siliconflow_base_url, s.siliconflow_api_key, s.siliconflow_model_fast

    if not s.deepseek_api_key:
        raise AgentError(ErrorCode.LLM_ERROR, "BANMA_DEEPSEEK_API_KEY 未配置（strong 档需要）")
    return "deepseek", s.deepseek_base_url, s.deepseek_api_key, s.deepseek_model_strong


def _build_model(tier: Tier, purpose: Purpose) -> ChatOpenAI:
    provider, base_url, api_key, model = _provider_of(tier)
    # 两家都默认开启思考模式，必须显式关闭，否则：
    #   · DeepSeek 官方：思考模式不支持 tool_choice，结构化输出直接 400
    #     "Thinking mode does not support this tool_choice"
    #   · 硅基 Qwen3.5：思考会白烧 token（实测 309 token 里 307 是思考，耗时 90s）
    # 另外思考模式下 temperature 会被忽略，关掉后才真正生效（温度分档才有意义）。
    # 注意字段名两家不同，不可混用。
    extra_body = (
        {"thinking": {"type": "disabled"}}
        if provider == "deepseek"
        else {"enable_thinking": False}
    )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=TEMPERATURE[purpose],
        timeout = get_settings().llm_timeout_s,
        max_retries=1,              # 升级策略由本模块负责，SDK这里只留一次
        extra_body=extra_body,
    )


def _map_upstream_error(exc: Exception, provider: str) -> AgentError:
    """上游异常 -> 契约错误码。带 provider 标签便于排障。"""
    if isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException)):
        return AgentError(ErrorCode.LLM_TIMEOUT, f"[{provider}] 上游模型超时")
    if isinstance(exc, openai.APIConnectionError):
        return AgentError(ErrorCode.LLM_ERROR, f"[{provider}] 无法连接上游模型")
    if isinstance(exc, openai.AuthenticationError):
        # 注意：这是"我们连上游的 Key 错"，不是 Java->Python 的 AUTH_FAILED
        return AgentError(ErrorCode.LLM_ERROR, f"[{provider}] 鉴权失败，请检查 API Key 或余额")
    if isinstance(exc, openai.APIStatusError):
        # 只回 "400" 等于没回：必须带上原始 detail 才能排障
        # （例：400 Thinking mode does not support this tool_choice）
        detail = str(getattr(exc, "message", "") or "").replace("\n", " ")[:200]
        return AgentError(
            ErrorCode.LLM_ERROR, f"[{provider}] 上游返回状态 {exc.status_code}: {detail}"
        )
    return AgentError(ErrorCode.LLM_ERROR, f"[{provider}] 上游调用失败: {exc}")


def _run(chain, messages, provider: str) -> Any:
    try:
        return chain.invoke(messages)
    except AgentError:
        raise
    except Exception as exc:
        raise  _map_upstream_error(exc, provider) from exc


def _run_with_escalation(tier_chain_factory, messages, purpose: Purpose) -> Any:
    """
    fast -> strong 显式降级；fast 档关闭时直连 strong。
    用显式 try/except 而非 with_fallbacks：后者不产生"降级已发生"事件，
    降级计数会失效
    tier_chain_factory的具体实现即_build_model
    """
    if not get_settings().fast_enabled:
        # fast 档默认关闭（ADR-014）：硅基免费 4B 实测首响应 71~90s 且系排队所致。
        # 强行走 fast 会先白等满 llm_timeout 再降级，比直连慢 20 倍。
        strong_provider, *_ = _provider_of(Tier.STRONG)
        return _run(tier_chain_factory(Tier.STRONG, purpose), messages, strong_provider)

    fast_provider, *_ = _provider_of(Tier.FAST)         # 配置缺失应 fail-fast，不触发降级

    try:
        return _run(tier_chain_factory(Tier.FAST, purpose), messages, fast_provider)
    except AgentError as exc:
        # 只有"上游不可用"类错误才升级；业务性错误（如解析）不在这里处理
        if exc.code not in (ErrorCode.LLM_TIMEOUT, ErrorCode.LLM_ERROR):
            raise
        _record_fallback(f"fast_failed_{exc.code.value}")
        strong_provider, *_ = _provider_of(Tier.STRONG)         # 只有真要降级时才要求 strong Key
        return _run(tier_chain_factory(Tier.STRONG, purpose), messages, strong_provider)


def structured_call(
        schema: type[BaseModel],
        messages: list[BaseMessage],
        purpose: Purpose,
) -> BaseModel:
    """结构化输出：异常级降级 + 解析级降级，两道防线。"""
    kwargs = {"method": "function_calling", "include_raw": True}

    def factory(tier: Tier, use_purpose: Purpose):
        return _build_model(tier, use_purpose).with_structured_output(schema, **kwargs)

    if purpose in FAST_ALLOWED:
        result = _run_with_escalation(factory, messages, purpose)
    else:
        strong_provider, *_ = _provider_of(Tier.STRONG)
        result = _run(factory(Tier.STRONG, purpose), messages, strong_provider)

    # 第二道：解析级降级（异常级管不到"返回了但格式不对"）
    if result.get("parsed") is None:
        _record_fallback("parse_failed_escalate_strong")
        strong_provider, *_ = _provider_of(Tier.STRONG)
        result = _run(factory(Tier.STRONG, purpose), messages, strong_provider)

    parsed = result.get("parsed")
    if parsed is None:
        raise AgentError(
            ErrorCode.STRUCTURE_PARSE_ERROR,
            f"结构化输出解析失败: {result.get('parsing_error')}",
        )
    return parsed


def chat_text(messages: list[BaseMessage], purpose: Purpose = Purpose.CHAT) -> str:
    """纯文本对话（非结构化）"""
    def factory(tier: Tier, use_purpose: Purpose):
        return _build_model(tier, use_purpose)

    message = _run_with_escalation(factory, messages, purpose)
    return message.content

