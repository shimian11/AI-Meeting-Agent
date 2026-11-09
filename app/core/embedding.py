"""
Embedding 调用（硅基 bge-m3，1024 维）
"""

import httpx
from loguru import logger

from app.core.config import get_settings
from app.core.errors import AgentError, ErrorCode

BATCH_SIZE = 8      #每批次处理8条
EXPECTED_DIM = 1024

def embed_texts(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    """
    批量向量化，返回顺序与入参严格一致。
    空列表直接抛错、不发请求 —— 硅基对空输入返回 400（code=20015），
    误判成"向量库不可用"会让上层错误降级（ADR-003 实测记录）。
    """
    if not texts:
        raise AgentError(ErrorCode.VALIDATION_ERROR, "embedding 输入为空，拒绝发起请求")

    s = get_settings()
    url = f"{s.siliconflow_base_url.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {s.siliconflow_api_key}"}
    vectors: list[list[float]] = []

    with httpx.Client(timeout=30) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            resp = client.post(
                url,
                headers=headers,
                json={"model": s.siliconflow_model_embed, "input": batch},
            )
            if resp.status_code != 200:
                raise AgentError(
                    ErrorCode.LLM_ERROR,
                    f"embedding 调用失败 HTTP {resp.status_code}: {resp.text[:200]}",
                )
            data = resp.json()["data"]
            # 按 index 排序：上游不保证返回顺序与入参一致，错配是静默致命的
            batch_vecs = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

            dim = len(batch_vecs[0])
            if dim != EXPECTED_DIM:
                raise AgentError(
                    ErrorCode.LLM_ERROR,
                    f"向量维度异常：{dim} != {EXPECTED_DIM}（须重建 collection）",
                )
            vectors.extend(batch_vecs)
            logger.debug(f"embedding 批次 {i // batch_size + 1}：{len(batch)} 条")

    return vectors