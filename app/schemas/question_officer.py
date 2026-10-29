"""出题官契约（逐字段对齐 docs/server/03 + contracts/question_officer_*.json）"""
from typing import Literal

from pydantic import BaseModel


class QuestionOfficerRequest(BaseModel):
    """Java端 -》 python的出题请求体"""
    trace_id: str  # 必填：Java Single-flight 生成透传
    session_id: str  # 必填：面试会话 ID
    user_input: str  # 必填：用户本轮话术
    resume_url: str | None = None  # 可空：MinIO 预签名 URL（未传简历 = null）
    direction_default: str = "后端开发"  # 可选：无简历时的兜底方向
    history_digest: list[str] = []  # 可选：历史轮次摘要（出题去重用）

class QuestionOfficerResponse(BaseModel):
    """Python → Java 的出题响应体"""
    trace_id: str
    intent: Literal["GENERATE", "SMALL_TALK"]
    question: list[str] = []
    resume_score: int | None = None
    resume_suggest: list[str] = []
    resume_type: str | None = None
    question_source: Literal["kb+resume", "resume_only", "none"]
    small_talk: str | None = None
    steps_trace: dict = {}