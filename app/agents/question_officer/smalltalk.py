"""
出题官 step6：兜底闲聊
Purpose.CHAT（温度 0.7，FAST_ALLOWED 含 CHAT——fast 开启时先打 fast）。
纯文本输出（非结构化）：闲聊话术不需要 schema 约束。
"""
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

from app.core.llm import chat_text, Purpose
from app.core.prompts import SMALL_TALK_SYSTEM_PROMPT
from app.core.trace import write_span, get_trace_id


def fallback_chat(user_input: str) -> str:
    """把闲聊引导回面试。失败会原样上抛（LLM_* 错），不编造话术。"""
    reply = chat_text(
        messages=[
            SystemMessage(content=SMALL_TALK_SYSTEM_PROMPT),
            HumanMessage(content=user_input)
        ],
        purpose=Purpose.CHAT,
    )
    write_span("agent.smalltalk", trace_id=get_trace_id(), chars=len(reply))
    logger.info(f"smalltalk chars={len(reply)}")
    return reply