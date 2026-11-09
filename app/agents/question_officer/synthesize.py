"""
出题官 step5：合成出题
输入三路材料：知识库片段 + 简历文本 + 历史摘要（Java 传入），
一次结构化调用生成约 10 条混合题。Purpose.GENERATE（0.5、直连 strong）。
"""
import time
from itertools import count

from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger
from pydantic import BaseModel, Field

from app.core.llm import structured_call, Purpose
from app.core.prompts import SYNTHESIZE_SYSTEM_PROMPT
from app.core.trace import write_span, get_trace_id

MAX_RESUME_CHARS = 6000

class GeneratedQuestions(BaseModel):
    """合成结果 schema：约 10 条"""
    questions: list[str] = Field(min_length=1, max_length=15)

def _format_kb(hits: list[dict]) -> str:
    """知识库片段格式化：空列表返回占位说明（prompt 里明确告知没有 kb 材料）。"""
    if not hits:
        return "（本次无知识库检索结果，请完全基于简历内容出题）"
    return "\n\n".join(f"【知识库片段{i + 1}】\n{h.get('text', '')}"
                       for i, h in enumerate(hits))

def _format_history(history_digest: list[str]) -> str:
    if not history_digest:
        return "（本轮无历史题目）"
    return "\n".join(f"- {item}" for item in history_digest)

def synthesize_questions(kb_hits: list[dict], resume_text: str | None,
                         direction: str, history_digest: list[str]) -> list[str]:
    """合成约 10 条题。kb_hits 为空 + 无简历 = 裸生成（凭模型知识出方向题）。"""
    start = time.perf_counter()
    resume_part = (resume_text or "")[:MAX_RESUME_CHARS]

    user_content = (
        f"目标方向：{direction}\n\n"
        f"== 知识库检索结果 ==\n{_format_kb(kb_hits)}\n\n"
        f"== 候选人简历 ==\n{resume_part or '（无简历）'}\n\n"
        f"== 历史已出题目（禁止重复） ==\n{_format_history(history_digest)}"
    )
    result: GeneratedQuestions = structured_call(
        schema=GeneratedQuestions,
        messages=[
            SystemMessage(content=SYNTHESIZE_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ],
        purpose=Purpose.GENERATE,
    )
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    write_span("agent.synthesize", trace_id=get_trace_id(),
               count=len(result.questions), duration_ms=duration_ms)
    logger.info(f"synthesize count={len(result.questions)} duration_ms={duration_ms}")
    return result.questions
