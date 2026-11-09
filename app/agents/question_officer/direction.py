"""
出题官 step3：方向归一化。四级来源循环（2026-11-08 定稿）：
（1）resume_type：关键词命中 → 归一；未命中 → LLM 归一；
（2）direction_default：合法值透传（非法值不透传，降级下一来源）；
（3）user_input：关键词命中 → 归一；未命中 → LLM 归一（话术无方向输出 NONE）；
（4）全落空 → VALIDATION_ERROR。
宁可显式失败，不编造数据——错方向会静默污染答题/评分/报告全链路。
"""
import time
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger
from pydantic import BaseModel

from app.core.errors import AgentError, ErrorCode
from app.core.llm import structured_call, Purpose
from app.core.prompts import DIRECTION_SYSTEM_PROMPT
from app.core.trace import write_span, get_trace_id

# 4 个合法方向：与 Qdrant payload 的 direction 过滤值、契约 direction_default 枚举逐字一致
DIRECTIONS = ("后端开发", "前端开发", "物联网开发", "大数据")
Direction = Literal["后端开发", "前端开发", "物联网开发", "大数据"]

# 本地关键词映射：小写包含匹配，命中就不进 LLM
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "后端开发": ("java", "spring", "jvm", "mysql", "redis", "后端", "服务端"),
    "前端开发": ("vue", "react", "javascript", "typescript", "前端", "web"),
    "物联网开发": ("物联网", "嵌入式", "stm32", "mqtt", "iot"),
    "大数据": ("hadoop", "spark", "flink", "hive", "大数据", "数仓"),
}

class DirectionOutput(BaseModel):
    """LLM 归一化 schema：NONE = 该文本没有指向任何方向（不强迫瞎选）。"""
    direction: Direction | Literal["NONE"]

def _llm_normalize(text: str, source: str) -> str | None:
    """LLM 语义归一。返回方向，或 None（NONE / 解析失败 → 降级下一来源）。
    服务不可用（LLM_TIMEOUT / LLM_ERROR）直接 raise——后续来源同样依赖模型，
    再试无意义，提前止损。
    """
    start = time.perf_counter()
    try:
        result: DirectionOutput = structured_call(
            schema=DirectionOutput,
            messages=[
                SystemMessage(content=DIRECTION_SYSTEM_PROMPT),
                HumanMessage(content=text),
            ],
            purpose=Purpose.EVAL,
        )
    except AgentError as exc:
        if exc.code is ErrorCode.STRUCTURE_PARSE_ERROR:
            # 关键行：模型两次都给不出合法枚举 → 降级下一来源，不硬兜底
            logger.warning(f"direction LLM 归一解析失败 source={source} text={text!r}，降级下一来源")
            return None
        raise

    if result.direction == "NONE":
        # 关键行：话术/描述里没有方向信息 → 降级下一来源，绝不拿 NONE 当方向用
        logger.info(f"direction LLM 判定无方向 source={source} text={text!r}，降级下一来源")
        return None

    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    write_span("agent.direction", trace_id=get_trace_id(),
               direction=result.direction, duration_ms=duration_ms)
    logger.info(f"direction={result.direction} reason=llm source={source} duration_ms={duration_ms}")
    return result.direction

def normalize_direction(resume_type: str | None, direction_default: str | None,
                        user_input: str) -> str:
    """四级语义链：简历推导 → 用户声明 → 话题归一 → 显式失败。禁止硬兜底。"""
    # (来源名, 候选文本)；None/空白 = 该来源缺位，跳过
    sources = (("resume_type", resume_type), ("direction_default", direction_default),
               ("user_input", user_input))
    for source, candidate in sources:
        if not candidate or not candidate.strip():
            continue
        text = candidate.lower()

        # ① 关键词命中 → 零成本归一
        for direction, words in _KEYWORDS.items():
            if any(w in text for w in words):
                logger.info(f"direction={direction} source={source} reason=keyword candidate={candidate!r}")
                return direction

        # ② default 来源：只透传合法值；非法值不透传（防 Qdrant 静默空召回）也不进 LLM
        if source == "direction_default":
            logger.warning(f"direction_default={candidate!r} 不在四方向枚举内，尝试下一来源")
            continue

        # ③ 关键词未命中 → LLM 语义归一（NONE/解析失败降级下一来源，服务异常上抛）
        llm_hit = _llm_normalize(candidate, source)
        if llm_hit:
            return llm_hit
        # None = 该来源给不出方向 → 落下一来源

    raise AgentError(
        ErrorCode.VALIDATION_ERROR,
        f"方向归一失败: resume_type={resume_type!r}, default={direction_default!r}，"
        f"请指定方向，合法方向={list(DIRECTIONS)}",
    )