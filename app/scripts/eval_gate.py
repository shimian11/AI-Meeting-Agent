"""5F-a 评测门禁：意图准确率 + 结构合法率。

用法：
    python -m app.scripts.eval_gate intent              # 意图全集（约 2.5 分钟）
    python -m app.scripts.eval_gate intent --limit 10   # 冒烟
    python -m app.scripts.eval_gate structure           # 结构合法率（10 条，约 80 秒）

════════════════════ 测什么 / 为什么 / 抓不到什么 ════════════════════
【意图准确率】
  被测代码：app/agents/question_officer/intent.py::classify_intent
           （Purpose.EVAL，温度 0.2，输出 Literal["GENERATE","SMALL_TALK"]）
  判定    ：expected == actual 的比例 ≥ 95%（docs/server/04 §2.2）
  为什么  ：意图判错会让后续五步全跑偏——该出题却走闲聊（用户没题）、
           该闲聊却触发简历解析（多花 3 次 LLM 且可能报错）。
  失败说明：prompt（core/prompts.py::INTENT_SYSTEM_PROMPT）规则覆盖不足，
           典型是"继续""再来几道"这类省略主语的短句被误判成 SMALL_TALK。

【结构合法率】
  被测代码：app/agents/question_officer/officer.py::run_question_officer
           + app/schemas/question_officer.py::QuestionOfficerResponse
  判定    ：返回体能通过 Pydantic 校验的比例 ≥ 99%（含 1 次重试后，server/04 §2.2）
  为什么  ：Java 侧按契约反序列化，字段缺失/类型错会直接 500 且日志里看不出来龙去脉；
           99% 的门禁意味着 100 次调用最多允许 1 次失败。
  失败说明：`STRUCTURE_PARSE_ERROR` = 模型输出反复不合 schema（改 prompt 或加 few-shot）；
           其他错误码（LLM_TIMEOUT / QDRANT_UNAVAILABLE）是**基础设施问题**，不算法不合法，
           会单独标注但同样让门禁失败（因为那次调用没产出可用结果）。

【两者共同抓不到】
- 题目质量：是否"好题" — 需人工 + 5F-b
- RAG 召回质量：依赖真实题库 — 5F-b（待 7C4 灌数据）
- 长尾/对抗输入（提示注入、超长文本）— 需单独建集，本脚本不覆盖
- 评分一致性（同答案 3 次极差 ≤10 分）— 需 score_set.jsonl，另建
══════════════════════════════════════════════════════════════════════
"""
import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

INTENT_SET_DEFAULT = Path("tests/evals/intent_set.jsonl")
INTENT_THRESHOLD = 0.95
STRUCTURE_THRESHOLD = 0.99

# 结构合法率用例：覆盖四方向 + 无方向 + 短句
STRUCTURE_CASES = [
    ("帮我出几道后端题", "后端开发"),
    ("我想练练前端", "前端开发"),
    ("物联网方向的题来几道", "物联网开发"),
    ("大数据面试题", "大数据开发"),
    # 注意：不传简历时必须给方向。无简历 + 无方向会（正确地）抛 VALIDATION_ERROR，
    # 那是输入校验行为，不属于"输出形状"问题，不该进结构合法率的分母
    ("帮我准备后端面试", "后端开发"),
    ("再来几道", "后端开发"),
    ("继续", "前端开发"),
    ("练练手", "物联网开发"),
    ("你好", "后端开发"),
    ("出十道题", "后端开发"),
]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        sys.exit(f"[错误] 评测集不存在: {path}")
    rows: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            sys.exit(f"[错误] 第 {i} 行不是合法 JSON: {e}")
    if not rows:
        sys.exit(f"[错误] 评测集为空: {path}")
    return rows


def eval_intent(path: Path, limit: int | None) -> bool:
    from app.agents.question_officer.intent import classify_intent

    rows = _load_jsonl(path)
    if limit:
        rows = rows[:limit]

    ok = 0
    failures: list[tuple[str, str, str, str]] = []

    for n, row in enumerate(rows, 1):
        expected = row["expected"]
        try:
            actual = classify_intent(row["input"]).intent
        except Exception as e:  # noqa: BLE001
            failures.append((row["id"], row["input"], expected, f"<异常 {type(e).__name__}>"))
            print(f"  [{n}/{len(rows)}] {row['id']} ✗ 异常 {type(e).__name__}: {e}")
            continue

        if actual == expected:
            ok += 1
            print(f"  [{n}/{len(rows)}] {row['id']} ✓ {actual}")
        else:
            failures.append((row["id"], row["input"], expected, actual))
            print(f"  [{n}/{len(rows)}] {row['id']} ✗ 期望 {expected}，实际 {actual}")

    acc = ok / len(rows)
    print(f"\n意图准确率 {acc:.1%}（{ok}/{len(rows)}），阈值 {INTENT_THRESHOLD:.0%}")

    if failures:
        print("\n失败清单（改 prompt 的输入）：")
        for fid, text, exp, act in failures:
            print(f"  {fid}  {text}  →  期望 {exp}，实际 {act}")
    return acc >= INTENT_THRESHOLD


def eval_structure(limit: int | None) -> bool:
    from app.agents.question_officer.officer import run_question_officer
    from app.core.errors import AgentError, ErrorCode
    from app.schemas.question_officer import QuestionOfficerRequest

    cases = STRUCTURE_CASES[:limit] if limit else STRUCTURE_CASES
    ok = 0
    failures: list[tuple[str, str]] = []

    for n, (text, direction) in enumerate(cases, 1):
        req = QuestionOfficerRequest(
            trace_id=f"eval-{uuid4().hex[:8]}",
            session_id=f"eval-{uuid4().hex[:8]}",
            user_input=text,
            resume_url=None,              # 不传简历：走纯题库路径，返回形状不变
            direction_default=direction,
        )
        try:
            resp = run_question_officer(req)
            ok += 1
            print(
                f"  [{n}/{len(cases)}] ✓ {resp.intent} / {len(resp.questions)} 题 / "
                f"source={resp.questions_source}"
            )
        except AgentError as e:
            failures.append((text, f"{e.code}"))
            print(f"  [{n}/{len(cases)}] ✗ {e.code}: {e.message}")
        except Exception as e:  # noqa: BLE001
            failures.append((text, f"{type(e).__name__}"))
            print(f"  [{n}/{len(cases)}] ✗ {type(e).__name__}: {e}")

    rate = ok / len(cases)
    print(f"\n结构合法率 {rate:.1%}（{ok}/{len(cases)}），阈值 {STRUCTURE_THRESHOLD:.0%}")
    if failures:
        print("\n失败清单：")
        for text, reason in failures:
            print(f"  {text}  →  {reason}")
    return rate >= STRUCTURE_THRESHOLD


def main() -> None:
    ap = argparse.ArgumentParser(description="5F-a 评测门禁（真实 LLM，会花钱）")
    ap.add_argument("suite", choices=["intent", "structure"], help="评测项")
    ap.add_argument("--limit", type=int, help="只跑前 N 条（冒烟用）")
    ap.add_argument("--set", default=str(INTENT_SET_DEFAULT), help="意图评测集路径")
    args = ap.parse_args()

    if args.suite == "intent":
        passed = eval_intent(Path(args.set), args.limit)
    else:
        passed = eval_structure(args.limit)

    print("\n" + ("✅ 门禁通过" if passed else "❌ 门禁未通过，阻断发布"))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
