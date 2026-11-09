"""题库入库接口：POST /v1/knowledge/ingest"""
from fastapi import APIRouter, Depends

from app.api.deps import require_token
from app.core.ingest import ingest_document
from app.schemas.knowledge import KnowledgeIngestResponse, KnowledgeIngestRequest

router = APIRouter(dependencies=[Depends(require_token)])

@router.post("/knowledge/ingest", response_model=KnowledgeIngestResponse)
def ingest(req: KnowledgeIngestRequest) -> KnowledgeIngestResponse:
    """
    注意是 `def` 不是 `async def`。
    内部全是阻塞 IO（httpx 下载、embedding 调用、Qdrant upsert）。写成同步函数后
    FastAPI 会自动丢到线程池执行，不阻塞事件循环。若写成 async def 却调同步代码，
    一个入库请求（3~20 秒）会把整个服务堵死。
    """
    return ingest_document(req)