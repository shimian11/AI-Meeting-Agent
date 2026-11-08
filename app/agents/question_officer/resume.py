"""
出题官 step2：简历解析评分
链路：预签名 URL 拉字节 → 20MB 上限 → pypdf 提文本层 → 体检 → 结构化评分。
 Purpose.SCORE 永远直连 strong
"""
import io
import time

import httpx
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger
from pydantic import BaseModel, Field
from pypdf import PdfReader

from app.core.errors import AgentError
from app.core.errors import ErrorCode
from app.core.llm import structured_call, Purpose
from app.core.prompts import RESUME_ANALYSIS_SYSTEM_PROMPT
from app.core.trace import write_span, get_trace_id

MAX_RESUME_BYTES = 20 * 1024 * 1024
MIN_TEXT_CHARS = 50

class ResumeAnalysis(BaseModel):
    """TODO step2中间产物，后面装配时映射到QuestionOfficerResponse"""
    score: int = Field(ge=0, le=100)
    resume_suggest: list[str]
    resume_type: str
    rag_query: list[str]
    resume_text: str = ""           # 合成简历定制题需要原文；

def _fetch_resume(resume_url: str) -> bytes:
    """拉取简历字节。Content-Length 预检 + 实际长度复查，双保险"""
    try:
        resp = httpx.get(resume_url, timeout=30.0, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise AgentError(ErrorCode.VALIDATION_ERROR, f"简历下载失败: {exc}") from exc
    if resp.status_code != 200:
        raise AgentError(ErrorCode.VALIDATION_ERROR, f"简历下载返回 {resp.status_code}")
    if len(resp.content) > MAX_RESUME_BYTES:
        raise AgentError(ErrorCode.VALIDATION_ERROR, "简历超过 20MB 上限")
    return resp.content

def _pdf_to_text(data: bytes) -> str:
    """pypdf 只提文本层；加密/损坏 PDF → VALIDATION_ERROR"""
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "")for page in reader.pages)
    except Exception as exc:
        raise AgentError(ErrorCode.VALIDATION_ERROR, f"PDF 解析失败: {exc}") from exc

def analyze_resume(resume_url: str) -> ResumeAnalysis:
    """有简历时的 step2 全链路；无简历（resume_url=None）由装配层跳过本函数。"""
    start = time.perf_counter()

    data = _fetch_resume(resume_url)
    text = _pdf_to_text(data)
    if len(text.strip()) < MIN_TEXT_CHARS:
        # 扫描件/纯图片简历：喂给 LLM 只会得到胡编的分数，早失败
        raise AgentError(ErrorCode.VALIDATION_ERROR,"简历文本层过短（疑似扫描件），无法评分")

    result: ResumeAnalysis = structured_call(
        schema=ResumeAnalysis,
        messages=[
            SystemMessage(content=RESUME_ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=f"简历全文如下：\n{text}"),
        ],
        purpose=Purpose.SCORE,
    )

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    write_span("agent.resume", trace_id=get_trace_id(),
               score=result.score, duration_ms=duration_ms)
    logger.info(f"resume score={result.score} type={result.resume_type} duration_ms={duration_ms}")
    return ResumeAnalysis(score=result.score, resume_suggest=result.resume_suggest,
                          resume_type=result.resume_type, rag_query=result.rag_query,
                          resume_text=text)
