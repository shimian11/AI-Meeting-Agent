"""RAG 冒烟：验证 search() 的检索链路、方向过滤、返回形状。

用法：uv run python -m app.scripts.rag_smoke

════════════════════ 断言 ↔ 被测代码 对照 ════════════════════

【断言 1】检索链路通：写入的点能被查回来
  被测代码（app/core/rag.py::RagRetriever.search）:
      result = self._client.query_points(
          collection_name=COLLECTION_NAME,   # 与 ingest 必须是同一个库
          query=query_vector,                # 查询用的就是写入时的同一个向量
          limit=top_k,                       # 5
          score_threshold=0.1,               # 自己对自己 cosine=1.0 > 0.1，不会被筛掉
          ...
      )
  断言:
      assert isinstance(hits, list) and any(h.get("question") == BACK_Q for h in hits)
  对应关系:
      写入时 vector=vec_back + payload.question=BACK_Q；查询时 query=vec_back。
      同一向量余弦相似度=1.0，必过 0.1 阈值 → 必能召回。
  失败说明（哪行可能坏了）:
      - upsert 没写进去（collection 名不一致、写库异常被吞）
      - create_collection 的 size=VECTOR_DIM 与实际向量长度不一致
      - score_threshold 被改大到 >1.0

【断言 2】方向过滤下推：返回结果不混其他方向
  被测代码:
      query_filter=Filter(
          must=[FieldCondition(key="direction", match=MatchValue(value=direction))]
      )
  断言:
      assert all(h.get("direction") == "后端开发" for h in hits)
  对应关系:
      payload 写入时是 {"direction": "后端开发"}；过滤条件 key="direction"、值为
      direction 参数。若 filter 未传或字段名不符，前端那条会混入，all() 立刻失败。
  失败说明:
      - query_filter 参数被删或未传入
      - payload 字段名与过滤 key 不一致（direction / dir）
      - 注意：create_payload_index 的 field_name 与过滤 key 不一致**不会**让本断言失败，
        只会让过滤退化为全表扫描（性能问题，本轮抓不到）

【断言 3】方向隔离：高相似度 + 错误方向 → 必须查不到
  被测代码: 同断言 2 的 query_filter，但 direction 传「前端开发」
  断言:
      assert not any(h.get("question") == BACK_Q for h in hits)
  对应关系:
      查询向量 vec_back 对后端题 score=1.0。若 query_filter 失效（被注释、参数名写错），
      会退化为全库检索 → 必然命中 BACK_Q → 断言失败。这是「必然失败场景」，
      比断言 2 更强：断言 2 在空结果时会假通过（all() 作用于空集合恒为真）。
  失败说明: query_filter 完全未生效（最严重形态）

【断言 4】返回形状：list 且每条带 score
  被测代码:
      points = result.points
      scores = [round(p.score, 3) for p in points]      # 属性名必须是 score
      hits = [{"score": p.score, **(p.payload or {})} for p in points]
      ...
      return hits                                        # 必须在 if not hits 块之外
  断言:
      assert isinstance(hits, list)
      assert all("score" in h for h in hits)
  对应关系:
      若 return 写在 `if not hits:` 内，有命中时函数走完没有 return → 返回 None →
      isinstance 失败；若属性写成 p.socre，抛 AttributeError 被 except 吞掉 → 返回 [] →
      all() 在空列表上为 True（假通过）。所以本条必须与断言 1 的「非空」配合才有意义。
  失败说明:
      - return hits 缩进在 if 块内 → 返回 None
      - 属性拼写错（p.socre）→ 静默空返回

【抓不到什么（测试边界）】
- payload 索引是否生效：索引只影响性能不影响结果，2 条数据测不出差异
  → 4C 灌千条后看日志里的 ms 是否随数据量线性增长
- 真实语义召回质量：随机向量无语义，score 只有 1.0 与 ~0 两个值
  → 5F 评测集（Recall@5）
- 阈值 0.1 是否合理：随机向量分布不真实（真实中文同主题 0.4~0.8）
  → 5F 看真实 score 分布定分位点
- 降级路径：脚本连的是可用 Qdrant
  → 需手动 stop 容器再跑，应返回 [] 且日志有 warning
- ingest 幂等与切分质量：不经过 ingest → 4C 单独验
═════════════════════════════════════════════════════════════
"""
import random

from qdrant_client.models import PointIdsList, PointStruct

from app.core.rag import COLLECTION_NAME, VECTOR_DIM, get_retriever

TEST_IDS = [990001, 990002]        # 高位 id，避开真实数据
BACK_Q = "幂等设计的几种实现"
FRONT_Q = "虚拟 DOM 的 diff 策略"


def main() -> None:
    r = get_retriever()
    r.ensure_collection()
    client = r._client

    # 只删自己的测试点，绝不清库（避免误删真实题库）
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=TEST_IDS),
    )

    vec_back = [random.random() for _ in range(VECTOR_DIM)]
    vec_front = [random.random() for _ in range(VECTOR_DIM)]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=TEST_IDS[0], vector=vec_back,
                        payload={"direction": "后端开发", "question": BACK_Q}),
            PointStruct(id=TEST_IDS[1], vector=vec_front,
                        payload={"direction": "前端开发", "question": FRONT_Q}),
        ],
    )

    # ── 断言 1 + 2 + 4 ──
    hits = r.search(vec_back, direction="后端开发")
    assert isinstance(hits, list), f"search 应返回 list，实际 {type(hits).__name__}"      # 断言 4
    assert any(h.get("question") == BACK_Q for h in hits), f"未召回测试点: {hits}"          # 断言 1
    assert all(h.get("direction") == "后端开发" for h in hits), f"方向过滤失效: {hits}"     # 断言 2
    assert all("score" in h for h in hits), f"hit 缺 score 字段: {hits}"                   # 断言 4
    print(f"✅ 检索链路通：后端向量 + 后端过滤 → {len(hits)} 条，全为后端题且带 score")

    # ── 断言 3 ──
    # 必须用「库里真实存在的其他方向」：若用不存在的方向，空结果会让断言退化为假通过
    hits = r.search(vec_back, direction="前端开发")
    assert not any(h.get("question") == BACK_Q for h in hits), f"方向隔离失效: {hits}"
    print("✅ 方向隔离：后端向量 + 前端过滤 → 未命中后端题")

    print("🎉 RAG 冒烟通过")


if __name__ == "__main__":
    main()
