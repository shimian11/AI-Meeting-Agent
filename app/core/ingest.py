"""
题库入库流水线
五步：取字节 → 解析 → 切分 → 向量化 → upsert

三条设计约束：
- **取字节与处理字节分离**：API 走 source_url，CLI 走本地文件，共用 ingest_from_bytes
- **幂等**：point_id = md5(f"{doc_id}#{chunk_index}")[:16]，重复灌覆盖而非新增
- **空文本/空块 → VALIDATION_ERROR**，绝不报 QDRANT_UNAVAILABLE（契约失败语义）
"""
import hashlib
import io
import time
from dataclasses import dataclass

import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pypdf import PdfReader
from qdrant_client.http.models import PointStruct

from app.core.embedding import embed_texts
from app.core.errors import AgentError, ErrorCode
from app.core.rag import COLLECTION_NAME, get_retriever
from app.schemas.knowledge import KnowledgeIngestRequest, KnowledgeIngestResponse

CHUNK_SIZE = 800            #单位是字符
CHUNK_OVERLAP = 80
UPSERT_BATCH = 100

# 分隔符按优先级依次尝试：先切段落，再切句子，最后硬切
# 中文标点必须显式列出，否则默认只认英文的 \n 和空格
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)


@dataclass
class IngestResult:
    points: int
    collection: str
    duration_ms: int


def _extract_text(data: bytes, doc_name: str) -> str:
    """
    PDF 走 pypdf，其余按 UTF-8 文本读。解码用 replace 兜底，不因个别乱码整篇失败。
    需要注意的是pypdf只提取文字层
    """
    if doc_name.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            raise AgentError(ErrorCode.VALIDATION_ERROR, f"PDF 解析失败: {e}") from e
    return data.decode("utf-8", errors="replace")


def preview_chunks(data: bytes, doc_name: str) -> list[str]:
    """只解析 + 切分，供 CLI 的 --dry-run 使用（不调 embedding，省额度）。"""
    text = _extract_text(data, doc_name)
    return [c for c in _SPLITTER.split_text(text) if c.strip()]


def _point_id(doc_id: str, index: int) -> int:
    """确定性 id：md5 前 16 hex = 64 bit，正好落在 Qdrant 的 uint64 范围。"""
    raw = hashlib.md5(f"{doc_id}#{index}".encode()).hexdigest()[:16]
    return int(raw, 16)


def ingest_from_bytes(
        data: bytes,
        doc_name: str,
        direction: str,
        doc_id: str,
        source_key: str = "",
        collection: str = COLLECTION_NAME,
) -> IngestResult:
    """核心流水线：字节 → 解析 → 切分 → 向量化 → upsert。API 与 CLI 共用。"""
    t0 = time.perf_counter()

    #1、解析
    text = _extract_text(data, doc_name)
    if not text.strip():
        raise AgentError(ErrorCode.VALIDATION_ERROR, "解析后无可提取文本（扫描件或缺文本层？）")

    #2、切分
    chunks = [c for c in _SPLITTER.split_text(text) if c.strip()]
    if not chunks:
         raise AgentError(ErrorCode.VALIDATION_ERROR, "切分后无有效块")
    logger.info(f"ingest {doc_name}: {len(text)} 字 → {len(chunks)} 块")

    #3、向量化
    vectors = embed_texts(chunks)
    if len(vectors) != len(chunks):
        raise AgentError(
            ErrorCode.INTERNAL_ERROR, f"向量数({len(vectors)}) ≠ 块数({len(chunks)})"
        )

    #4、组装payload
    retriever = get_retriever()
    retriever.ensure_collection()
    client = retriever._client

    now = int(time.time())
    points = [
        PointStruct(
            id=_point_id(doc_id, i),
            vector=vec,
            payload={
                "doc_id": doc_id,
                "chunk_index": i,
                "direction": direction,
                "text": chunk,
                "source": doc_name,
                "source_key": source_key,
                "created_at": now,
            }
        )
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]

    #5、分批upsert
    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(collection_name=collection, points=points[start : start + UPSERT_BATCH])

    elapsed = int((time.perf_counter() - t0) * 1000)
    logger.info(f"ingest 完成 {doc_name}: {len(points)} 点 / {elapsed}ms")
    return IngestResult(points=len(points), collection=collection, duration_ms=elapsed)


def ingest_document(req: KnowledgeIngestRequest) -> KnowledgeIngestResponse:
    """API 入口：拉预签名 URL → 交给核心流水线。"""
    try:
        resp = httpx.get(req.source_url, timeout=60, follow_redirects=True)
    except httpx.HTTPError as e:
        raise AgentError(ErrorCode.VALIDATION_ERROR, f"下载失败: {e}") from e

    if resp.status_code != 200:
        # 预签名过期 / 权限不足都归 VALIDATION_ERROR，不得报向量库故障
        raise AgentError(
            ErrorCode.VALIDATION_ERROR,
            f"下载失败 HTTP {resp.status_code}（预签名过期或权限不足）",
        )

    result = ingest_from_bytes(
        data=resp.content,
        doc_name=req.doc_name,
        direction=req.direction,
        doc_id=req.doc_id,
        source_key=req.source_key,
        collection=req.collection,
    )
    return KnowledgeIngestResponse(
        trace_id=req.trace_id,
        doc_id=req.doc_id,
        points=result.points,
        collection=result.collection,
        duration_ms=result.duration_ms,
    )


