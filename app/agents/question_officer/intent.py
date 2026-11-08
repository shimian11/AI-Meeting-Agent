"""
出题官意图路由
温度 0.2（Purpose.EVAL）、输出二选一枚举；解析失败抛 STRUCTURE_PARSE_ERROR
禁止兜底成 SMALL_TALK，宁可让上游重试
"""
import time
from typing import Literal

from loguru import logger
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.llm import structured_call, Purpose
from app.core.prompts import INTENT_SYSTEM_PROMPT
from app.core.trace import write_span, get_trace_id


class IntentOutput(BaseModel):
    """意图路由的 schema"""
    intent: Literal["GENERATE", "SMALL_TALK"]

def classify_intent(user_input: str) -> IntentOutput:
    """
    判断用户本轮话术：要出题（GENERATE）还是闲聊（SMALL_TALK）。
    消息只有两条：system（角色+规则）+ human（用户原话）。
    history_digest 不参与——那是出题去重用的，路由只看本轮话术。
    """
    start = time.perf_counter()

    result: IntentOutput = structured_call(
        schema=IntentOutput,
        messages=[
            SystemMessage(content=INTENT_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ],
        purpose=Purpose.EVAL,
    )
    duration_ms = round((time.perf_counter() - start) * 1000,1)

    # 可观测：span + 行内日志
    write_span("agent.intent", trace_id = get_trace_id(),
               intent=result.intent, duration_ms=duration_ms)
    logger.info(f"intent={result.intent} duration_ms={duration_ms}")
    return result

