"""题库入库契约模型"""
from typing import Literal

from pydantic import BaseModel

Direction = Literal["后端开发", "前端开发", "物联网开发", "大数据开发"]

class KnowledgeIngestRequest(BaseModel):
    trace_id: str
    doc_id: str
    source_url: str              # MinIO 预签名 URL，15 分钟有效
    direction: Direction
    doc_name: str
    source_key: str = ""        # MinIO 对象键（payload 溯源用）
    collection: str = "banma_questions"

class KnowledgeIngestResponse(BaseModel):
    trace_id: str
    doc_id: str
    points: int
    collection: str
    duration_ms: int