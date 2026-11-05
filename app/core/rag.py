"""
Qdrant 向量检索封装。
三条硬约束：
- 过滤字段必须建 payload 索引，否则 direction 过滤退化为全表扫描（不报错，只是慢且漏召回）
- 检索失败降级返回 []，不向上抛
- 日志必须带 hits / scores / ms
"""
import time
from functools import lru_cache

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from app.core.config import get_settings

COLLECTION_NAME = "banma_questions"
VECTOR_DIM = 1024

class RagRetriever:
    """出题官的题库检索器"""
    def __init__(self, client: QdrantClient) -> None:
        self._client = client
        self._ensured = False       #幂等键，判断是否创建好了collection和payload索引

    def ensure_collection(self) -> None:
        """幂等建 collection + payload 索引，惰性执行（首次检索前）。"""
        if self._ensured:
            return

        try:
            if not self._client.collection_exists(COLLECTION_NAME):
                self._client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                )
                logger.info(f"已创建 collection {COLLECTION_NAME}（{VECTOR_DIM} 维 / COSINE）")

            # 过滤字段建索引。不建则每次过滤扫全表，且【不报错】
            self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="direction",
                field_schema="keyword",
            )
            self._ensured = True
        except Exception as e:
            logger.warning(f"Qdrant 建表/建索引失败（本次检索降级）: {e}")

    def search(
            self,
            query_vector: list[float],
            direction: str,
            top_k: int=5,
    ) -> list[dict]:
        """payload 过滤下推检索；异常一律降级为空列表。"""
        self.ensure_collection()
        t0 = time.perf_counter()
        try:
            result = self._client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
                score_threshold=0.1,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="direction", match=MatchValue(value=direction)),
                    ]
                ),
            )
            points = result.points
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            scores = [round(p.score, 3) for p in points]
            hits = [{"score": p.score, **(p.payload or {})} for p in points]

            logger.info(
                f"RAG direction={direction} hits={len(hits)} ms={elapsed_ms} scores={scores}"
            )
            if not hits:
                logger.warning(
                    f"RAG 空召回 direction={direction}（依次查：库有数据？过滤条件对？阈值过高？）"
                )
            return hits
        except Exception as e:
            logger.warning(f"RAG 检索失败（降级 resume_only）: {e}")
            return []


@lru_cache
def get_retriever() -> RagRetriever:
    """模块级单例：全进程共用一个连接池。"""
    settings = get_settings()
    return RagRetriever(QdrantClient(url=settings.qdrant_url, timeout=5))