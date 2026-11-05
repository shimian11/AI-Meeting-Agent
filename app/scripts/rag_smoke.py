"""RAG 冒烟：灌 2 条假向量 → 验证检索链路与方向隔离。

用法：uv run python -m app.scripts.rag_smoke

【测什么 / 为什么】
1. 后端向量 + 后端过滤 → 必须召回刚写入的测试点
   测：写入→检索链路通。
   为什么：一条断言同时验证三件事 —— upsert 真的写进库、collection 维度(1024)与向量长度一致、
            相似度计算正常（自己对自己 score≈1.0，能过阈值 0.1）。
           抓：维度不匹配、collection 名写错、阈值设成 >1.0。

2. 返回结果的 direction 必须全为「后端开发」
   测：过滤下推生效，没有跨方向污染。
   为什么：query_filter 没传、payload 字段名写成 dir、MatchValue 值写错，都会让前端题混入。

3. 后端向量 + 前端过滤 → 不得命中后端题
   测：过滤真的在拦，而不是碰巧没返回。
   为什么：这是必然失败场景 —— 该向量对后端题的 score=1.0 远超阈值 0.1，过滤一旦失效必命中。
           它比第 2 条更强：第 2 条可能因「空结果」而假通过（all() 作用在空集合上恒为真）。

4. 返回类型是 list 且每条带 score 字段
   测：返回形状。
   为什么：抓两类会被 except 吞掉的 bug —— 漏写 return（返回 None）、属性拼写错（如 p.socre）。
           这两种都表现为「检索不到题」，从日志看不出是代码问题。

【抓不到什么】
- payload 索引是否生效：索引只影响性能不影响结果，2 条数据测不出差异 → 4C 灌千条后看 ms 日志
- 真实语义召回质量：随机向量无语义，score 只有 1.0 和 ~0 → 5F 评测集（Recall@5）
- 阈值 0.1 是否合理：随机向量分布不真实 → 5F 看真实 score 分布定分位点
- 降级路径：脚本连的是可用 Qdrant → 需手动 stop 容器再跑，应返回 [] 且日志有 warning
- ingest 幂等与切分质量：不经过 ingest → 4C 单独验
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

    # 断言 1 + 2 + 4
    hits = r.search(vec_back, direction="后端开发")
    assert isinstance(hits, list), f"search 应返回 list，实际 {type(hits).__name__}"
    assert any(h.get("question") == BACK_Q for h in hits), f"未召回测试点: {hits}"
    assert all(h.get("direction") == "后端开发" for h in hits), f"方向过滤失效: {hits}"
    assert all("score" in h for h in hits), f"hit 缺 score 字段: {hits}"
    print(f"✅ 检索链路通：后端向量 + 后端过滤 → {len(hits)} 条，全为后端题且带 score")

    # 断言 3：必须用「库里真实存在的其他方向」，否则空结果会让断言退化为假通过
    hits = r.search(vec_back, direction="前端开发")
    assert not any(h.get("question") == BACK_Q for h in hits), f"方向隔离失效: {hits}"
    print("✅ 方向隔离：后端向量 + 前端过滤 → 未命中后端题")

    print("🎉 RAG 冒烟通过")


if __name__ == "__main__":
    main()
