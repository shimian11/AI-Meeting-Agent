"""检索去重单测（mock embed 与 retriever，不真调、零花费）。

三份说明（AGENT.md §8）：

1. 测什么 —— 被测代码 `retrieval.py::retrieve_questions()` 的三段：
   - 查询兜底行 `queries = rag_query or [direction]`：断言
     `test_empty_query_falls_back_to_direction` 里 mock 收到的 embed 入参
     == ["后端开发"]（无简历时 rag_query 为空 → 用方向当查询词）；
   - 去重行 `_dedupe` 的 `key = f"{doc_id}#{chunk_index}"` + 保留最高分：
     `test_dedupe_keeps_highest_score`（两个查询词命中同一 chunk，score 0.9 与
     0.7 → 结果 1 条且 score==0.9）——若去掉前置的 `merged.sort(reverse=True)`，
     保住的会是 0.7 那条，本测试必失败；
   - 截断行 `[:MAX_TOTAL]`：`test_truncate_to_max_total`（构造 12 条不同 chunk
     → 结果 10 条）。
   另有 `test_embed_failure_degrades`：embed 抛异常 → 返回 []（降级 resume_only，
   对应 `except Exception` 块）。
2. 为什么测
   去重失败 = 同一题面重复出现在题单（用户可见的 bug）；降级失败 = 硅基一挂整个
   出题 500（违反"RAG 挂了走 resume_only"的韧性设计）；兜底查询缺失 = 不传简历
   的黄金路径直接空检索。
3. 抓不到什么
   mock 抓不到 embedding 质量（查询词向量准不准归 5F-b Recall@5）；抓不到
   Qdrant 真实过滤行为（4B 冒烟已覆盖）；抓不到跨 chunk 的语义重复（切分 overlap
   导致的近似重复由 step5 合成时 LLM 去重）。

Mock 位置：patch `app.agents.question_officer.retrieval` 命名空间的
`embed_texts` 与 `get_retriever`（后者返回桩 retriever，其 search 按入参向量
顺序返回预制 hits——embed 契约"返回顺序与入参严格一致"是本测试的依赖前提）。
"""
from app.agents.question_officer import retrieval as retrieval_mod
from app.agents.question_officer.retrieval import MAX_TOTAL, retrieve_questions


class _StubRetriever:
    """按向量顺序弹出预制 hits（模拟 search 的 score 降序返回）。"""

    def __init__(self, per_query: list[list[dict]]):
        self._per_query = per_query
        self.calls: list[dict] = []

    def search(self, query_vector, direction, top_k=5):
        self.calls.append({"direction": direction, "top_k": top_k})
        return self._per_query.pop(0)


def _hit(doc_id: str, chunk: int, score: float, text: str = "题面") -> dict:
    return {"score": score, "doc_id": doc_id, "chunk_index": chunk,
            "direction": "后端开发", "text": text}


def test_dedupe_keeps_highest_score(monkeypatch):
    hits_a = [_hit("d1", 0, 0.7)]
    hits_b = [_hit("d1", 0, 0.9)]          # 同 chunk，另一查询词命中更高分
    stub = _StubRetriever([hits_a, hits_b])
    monkeypatch.setattr(retrieval_mod, "embed_texts", lambda texts: [[0.1], [0.2]])
    monkeypatch.setattr(retrieval_mod, "get_retriever", lambda: stub)

    result = retrieve_questions(["并发", "JVM"], "后端开发")

    assert len(result) == 1                # 去重成功
    assert result[0]["score"] == 0.9       # 保的是高分那条（sort 前置的证明）


def test_empty_query_falls_back_to_direction(monkeypatch):
    stub = _StubRetriever([[_hit("d1", 0, 0.8)]])
    captured = {}
    monkeypatch.setattr(
        retrieval_mod, "embed_texts",
        lambda texts: captured.update(texts=texts) or [[0.1]],
    )
    monkeypatch.setattr(retrieval_mod, "get_retriever", lambda: stub)

    result = retrieve_questions([], "后端开发")     # 无简历 → rag_query 为空

    assert captured["texts"] == ["后端开发"]          # 兜底查询词生效
    assert stub.calls[0]["direction"] == "后端开发"   # direction 下推过滤


def test_truncate_to_max_total(monkeypatch):
    many = [_hit(f"d{i}", 0, 1.0 - i / 100) for i in range(12)]
    stub = _StubRetriever([many])
    monkeypatch.setattr(retrieval_mod, "embed_texts", lambda texts: [[0.1]])
    monkeypatch.setattr(retrieval_mod, "get_retriever", lambda: stub)

    result = retrieve_questions(["Java"], "后端开发")

    assert len(result) == MAX_TOTAL == 10


def test_embed_failure_degrades(monkeypatch):
    def boom(texts):
        raise RuntimeError("硅基流动不可达")

    monkeypatch.setattr(retrieval_mod, "embed_texts", boom)

    assert retrieve_questions(["Java"], "后端开发") == []   # 降级 []，走 resume_only
