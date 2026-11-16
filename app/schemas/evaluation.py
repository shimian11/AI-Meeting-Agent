"""评分官契约"""
from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    """Java -> Python的评分请求"""
    trace_id: str
    session_id: str
    question: str
    answer: str
    resume_digest: list[str] = []
    standard: list[str] = []

class EvaluationResponse(BaseModel):
    """Python -> Java的评分响应"""
    trace_id: str
    score: int
    feedback: str
    missing_points: list[str] = []
    follow_up_needed: bool = False
    follow_up_question: str | None = None

class EvaluationDraft(BaseModel):
    """模型的结构化输出：不含 trace_id（由服务端回填，防止模型编造）"""
    score: int = Field(ge=0, le=100, description="0~100 整数")
    feedback: str = Field(description="2~3句中文点评")
    missing_points: list[str] = Field(default_factory=list)
    follow_up_needed: bool = False
    follow_up_question: str | None = None