"""
出题官 step3：方向归一化。三情况：
（1）无简历/空描述 → direction_default 原样透传（须为 4 枚举之一，非法即拒）；
（2）本地关键词命中 → 直接归一（零成本）；
（3）关键词未命中 → LLM 归一（结构化输出锁 4 枚举）；失败 → 原样上抛。
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
    """LLM 归一化 schema：只许从 4 枚举选一。"""
    direction: Direction

def normalize_direction(resume_type: str | None, direction_default: str) -> str:
    """归一到 4 方向；彻底失败上抛 STRUCTURE_PARSE_ERROR，绝不硬兜底。"""
    #（1）无简历/空描述 -> default原样透传
    if not resume_type or not resume_type.strip():
        if direction_default not in DIRECTIONS:
            #透传枚举外值会让 Qdrant filter 命中 0 → 静默空召回，必须现在就炸
            raise AgentError(
                ErrorCode.VALIDATION_ERROR,
                f"direction_default={direction_default!r} 不在四方向枚举内",
            )
        logger.info(f"direction=default({direction_default}) reason=no_resume_type")
        return direction_default

    text = resume_type.lower()

    #（2）本地关键词命中 -> 零成本归一
    for direction, words in _KEYWORDS.items():
        if any(w in text for w in words):
            logger.info(f"direction={direction} reason=keyword resume_type={resume_type!r}")
            return direction

    #（3）关键词未命中 ->LLM 语义归一；失败原样上抛
    start = time.perf_counter()
    try:
        result: DirectionOutput = structured_call(
            schema=DirectionOutput,
            messages=[
                SystemMessage(content=DIRECTION_SYSTEM_PROMPT),
                HumanMessage(content=f"简历方向描述：{resume_type}"),
            ],
            purpose=Purpose.EVAL,
        )
    except AgentError:
        # 关键行：吞掉这里换成 return "后端开发" = test_llm_failure_raises_not_fallback 必红
        logger.warning(f"direction normalize failed resume_type={resume_type!r}")
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    write_span("agent.direction", trace_id=get_trace_id(),
               direction=result.direction, duration_ms=duration_ms)
    logger.info(f"direction={result.direction} reason=llm duration_ms={duration_ms}")
    return result.direction

