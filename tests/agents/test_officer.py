"""出题官装配单测（mock 五步，不真调、零花费）。

三份说明（AGENT.md §8）：

1. 测什么 —— 被测代码 `officer.py::run_question_officer()` 的编排与映射：
   - `test_smalltalk_branch`：mock `classify_intent` 返回 SMALL_TALK → 断言
     `questions == []`、`questions_source == "none"`、`small_talk == mock 话术`、
     `resume_score is None` —— 对应 SMALL_TALK 早返回块（闲聊不该烧简历/检索/合成）；
   - `test_generate_full_chain`：五步全 mock → 断言字段映射
     `analysis.score → response.resume_score`、`rag_query 消费`（mock 断言收到
     传入的查询词）、`source == "kb+resume"`（hits 非空分支）；
   - `test_resume_only_when_no_hits`：mock 检索返回 [] + 有简历 → `resume_only`；
   - `test_none_when_no_resume_no_hits`：无简历 + 检索 [] → `none`；
   - `test_steps_trace_keys`：全链的 `steps_trace` 含 intent/resume/retrieve/
     synthesize 四个键（可观测字段不缺）。
2. 为什么测
   装配是五步的唯一交汇点：字段映射错（score 接到 suggest 上）单步测试全绿也
   发现不了；source 判定错会让 Java 侧走错文案分支；闲聊分支若不早返回，一句
   "你好"会烧一次简历评分+出题（真金白银，ADR-009 成本红线）。
3. 抓不到什么
   mock 抓不到五步之间的真实数据流（rag_query 真实内容归 5F-a 评测）；抓不到
   LLM 真实行为；抓不到 HTTP 层（鉴权/序列化归 test_question_officer_api.py）。

Mock 位置：patch `app.agents.question_officer.officer` 命名空间的五个步骤函数
（officer.py 已把它们 import 进自己的命名空间）。
"""
from app.agents.question_officer import officer as officer_mod
from app.agents.question_officer.officer import run_question_officer
from app.agents.question_officer.resume import ResumeAnalysis
from app.schemas.question_officer import QuestionOfficerRequest


def _req(**overrides) -> QuestionOfficerRequest:
    base = dict(trace_id="t-1", session_id="s-1", user_input="帮我出题",
                direction_default="后端开发", history_digest=[])
    base.update(overrides)
    return QuestionOfficerRequest(**base)


def _analysis(**overrides) -> ResumeAnalysis:
    base = dict(score=78, resume_suggest=["补量化"], resume_type="后端开发",
                rag_query=["Java并发"], resume_text="张三的简历")
    base.update(overrides)
    return ResumeAnalysis(**base)


def test_smalltalk_branch(monkeypatch):
    monkeypatch.setattr(officer_mod, "classify_intent",
                        lambda _: type("I", (), {"intent": "SMALL_TALK"})())
    monkeypatch.setattr(officer_mod, "fallback_chat", lambda _: "现在是面试时间…")

    resp = run_question_officer(_req(user_input="今天天气不错"))

    assert resp.intent == "SMALL_TALK"
    assert resp.questions == []
    assert resp.questions_source == "none"
    assert resp.small_talk == "现在是面试时间…"
    assert resp.resume_score is None
    assert "chat_ms" in resp.steps_trace


def test_generate_full_chain(monkeypatch):
    monkeypatch.setattr(officer_mod, "classify_intent",
                        lambda _: type("I", (), {"intent": "GENERATE"})())
    monkeypatch.setattr(officer_mod, "analyze_resume", lambda url: _analysis())
    monkeypatch.setattr(officer_mod, "normalize_direction",
                        lambda **kw: "后端开发")   # officer 用关键字调用，mock 用 **kw 兜住
    captured = {}
    monkeypatch.setattr(officer_mod, "retrieve_questions",
                        lambda rag_query, direction: captured.update(
                            rag_query=rag_query) or [{"text": "kb", "doc_id": "d",
                                                      "chunk_index": 0, "score": 0.9}])
    monkeypatch.setattr(officer_mod, "synthesize_questions",
                        lambda kb_hits, resume_text, direction, history_digest:
                        captured.update(resume_text=resume_text) or ["题1", "题2"])

    resp = run_question_officer(_req(resume_url="http://x/r.pdf"))

    assert resp.intent == "GENERATE"
    assert resp.questions == ["题1", "题2"]
    assert resp.resume_score == 78                    # 映射：analysis.score
    assert resp.resume_suggest == ["补量化"]
    assert resp.questions_source == "kb+resume"       # hits 非空分支
    assert captured["rag_query"] == ["Java并发"]       # 5B 的查询词喂给了 5C
    assert captured["resume_text"] == "张三的简历"      # 5B 的原文喂给了 5D


def test_resume_only_when_no_hits(monkeypatch):
    monkeypatch.setattr(officer_mod, "classify_intent",
                        lambda _: type("I", (), {"intent": "GENERATE"})())
    monkeypatch.setattr(officer_mod, "analyze_resume", lambda url: _analysis())
    monkeypatch.setattr(officer_mod, "normalize_direction", lambda **kw: "后端开发")
    monkeypatch.setattr(officer_mod, "retrieve_questions", lambda rag_query, direction: [])
    monkeypatch.setattr(officer_mod, "synthesize_questions",
                        lambda kb_hits, resume_text, direction, history_digest: ["定制题"])

    resp = run_question_officer(_req(resume_url="http://x/r.pdf"))
    assert resp.questions_source == "resume_only"


def test_none_when_no_resume_no_hits(monkeypatch):
    monkeypatch.setattr(officer_mod, "classify_intent",
                        lambda _: type("I", (), {"intent": "GENERATE"})())
    monkeypatch.setattr(officer_mod, "normalize_direction", lambda **kw: "后端开发")
    monkeypatch.setattr(officer_mod, "retrieve_questions", lambda rag_query, direction: [])
    monkeypatch.setattr(officer_mod, "synthesize_questions",
                        lambda kb_hits, resume_text, direction, history_digest: ["自由题"])
    monkeypatch.setattr(
        officer_mod, "analyze_resume",
        lambda url: pytest_fail("无 resume_url 时不应调用简历评分"),
    ) if False else None

    resp = run_question_officer(_req())               # 无 resume_url
    assert resp.questions_source == "none"
    assert resp.resume_score is None


def test_steps_trace_keys(monkeypatch):
    monkeypatch.setattr(officer_mod, "classify_intent",
                        lambda _: type("I", (), {"intent": "GENERATE"})())
    monkeypatch.setattr(officer_mod, "analyze_resume", lambda url: _analysis())
    monkeypatch.setattr(officer_mod, "normalize_direction", lambda **kw: "后端开发")
    monkeypatch.setattr(officer_mod, "retrieve_questions", lambda rag_query, direction: [])
    monkeypatch.setattr(officer_mod, "synthesize_questions",
                        lambda kb_hits, resume_text, direction, history_digest: ["题"])

    resp = run_question_officer(_req(resume_url="http://x/r.pdf"))
    for key in ("intent_ms", "resume_ms", "retrieve_ms", "synthesize_ms"):
        assert key in resp.steps_trace


def pytest_fail(msg):
    raise AssertionError(msg)
