"""
出题官 step4：多查询检索 + 去重。
逐 rag_query 检索 → doc_id#chunk_index 去重保最高分 → 截断。
embed 失败降级 []：RAG 挂了走 resume_only，服务不挂。
"""
import time

from loguru import logger

from app.core.embedding import embed_texts
from app.core.rag import get_retriever
from app.core.trace import write_span, get_trace_id

TOP_K_PER_QUERY = 5                 #单查询topK
MAX_TOTAL = 10                      #去重后上限

def _dedupe(hits: list[dict]) -> list[dict]:
    """按 doc_id#chunk_index 去重，保留首次出现（各查询按 score 降序返回，先见分高）。"""
    seen: set[str] = set()
    unique: list[dict] = []
    for hit in hits:
        key = f"{hit.get('doc_id')}#{hit.get('chunk_index')}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique

def retrieve_questions(rag_query: list[str], direction: str) -> list[dict]:
    """多查询检索；返回 payload 列表（含 text/score）"""
    start = time.perf_counter()
    queries = rag_query or [direction]

    try:
        vectors = embed_texts(queries)
    except Exception as exc:
        logger.warning(f"embedding 失败，降级 resume_only: {exc}")
        write_span("agent.retrieve", trace_id=get_trace_id(), hits=0, degraded=True)
        return []

    retriever = get_retriever()
    merged: list[dict] = []
    for vector in vectors:
        merged.extend(retriever.search(query_vector=vector, direction=direction, top_k=TOP_K_PER_QUERY))

    #search 返回按 score 降序，跨查询合并后再按分排一次，去重才保得住最高分
    merged.sort(key=lambda h : h["score"], reverse=True)
    hits = _dedupe(merged)[:MAX_TOTAL]

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    write_span("agent.retrieve", trace_id=get_trace_id(),
               queries=len(queries),hits=len(hits),duration_ms=duration_ms)
    logger.info(f"retrieve direction={direction} queries={len(queries)} hits={len(hits)} ms={duration_ms}")
    return hits



