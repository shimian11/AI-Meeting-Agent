"""合成出题单测（mock LLM，不真调、零花费）。

三份说明（AGENT.md §8）：

1. 测什么 —— 被测代码 `synthesize.py::synthesize_questions()` 的 prompt 组装行
   `user_content = f"目标方向…{_format_kb(kb_hits)}…{resume_part}…{_format_history(...)}"`
   与透传行 `return result.questions`。断言（通过捕获 mock 收到的 messages）：
   - `test_kb_and_resume_in_prompt`：kb_hits 的 text 与 resume_text 都出现在
     HumanMessage 里（材料没丢）；
   - `test_empty_kb_placeholder`：kb_hits=[] 时 prompt 含"无知识库检索结果"占位
     ——防模型对着空段落幻觉（对应 `_format_kb` 空态分支）；
   - `test_resume_truncated`：6000+ 字简历 → prompt 里该段 ≤6000 字
     （对应 `[:MAX_RESUME_CHARS]` 截断行，防 token/费用失控）；
   - `test_history_digest_in_prompt`：历史摘要逐条出现在 prompt 且带"- "前缀
     （防重复出题的原材料丢失）；
   - `test_questions_passthrough_and_purpose`：返回值 == mock 的 questions 且
     `purpose is Purpose.GENERATE`（0.5、直连 strong——防止有人把它挪进
     FAST_ALLOWED，出题走免费 4B 违反 ADR-009）。
2. 为什么测
   合成是出题官的"总装车间"：材料拼装丢一路（kb/简历/历史）用户都会直接感知
   （题不对方向/重复出题/脱离简历）；截断缺失会在大简历上打爆 token 预算
   （ADR-009 ¥20/月红线）；purpose 断言守住出题永远走 strong。
3. 抓不到什么
   mock 抓不到真实出题质量（题面是否优质、是否真没重复——归 5F-a/5F-b 评测）；
   抓不到 tokens 统计（steps_trace 的 token 字段归 5E 装配汇总）。

Mock 位置：patch `app.agents.question_officer.synthesize` 命名空间的
`structured_call`，捕获 messages 做内容断言。
"""
from app.agents.question_officer import synthesize as syn_mod
from app.agents.question_officer.synthesize import MAX_RESUME_CHARS, GeneratedQuestions, synthesize_questions
from app.core.llm import Purpose


def _capture_llm(monkeypatch, questions: list[str]) -> dict:
    """桩 structured_call：捕获 (messages, purpose)，返回预制题目。"""
    captured = {}

    def fake_call(schema, messages, purpose):
        captured["messages"] = messages
        captured["purpose"] = purpose
        assert schema is GeneratedQuestions
        return GeneratedQuestions(questions=questions)

    monkeypatch.setattr(syn_mod, "structured_call", fake_call)
    return captured


def test_kb_and_resume_in_prompt(monkeypatch):
    captured = _capture_llm(monkeypatch, ["题1"])
    kb = [{"text": "HashMap 扩容机制是什么", "doc_id": "d1", "chunk_index": 0, "score": 0.9}]

    synthesize_questions(kb, "三年 Java 经验", "后端开发", [])

    human = captured["messages"][1].content
    assert "HashMap 扩容机制是什么" in human      # kb 材料在 prompt 里
    assert "三年 Java 经验" in human               # 简历材料在 prompt 里
    assert "后端开发" in human                     # 方向声明在 prompt 里


def test_empty_kb_placeholder(monkeypatch):
    captured = _capture_llm(monkeypatch, ["裸出题1"])

    synthesize_questions([], None, "前端开发", [])

    human = captured["messages"][1].content
    assert "无知识库检索结果" in human             # 空态占位，防幻觉
    assert "（无简历）" in human                   # 无简历占位


def test_resume_truncated(monkeypatch):
    captured = _capture_llm(monkeypatch, ["题1"])
    long_resume = "项目经历" * 5000                # 20000 字，远超上限

    synthesize_questions([], long_resume, "后端开发", [])

    human = captured["messages"][1].content
    resume_section = human.split("== 候选人简历 ==")[1].split("== 历史已出题目")[0]
    assert len(resume_section) <= MAX_RESUME_CHARS + 50   # 截断生效（+50 为模板余量）


def test_history_digest_in_prompt(monkeypatch):
    captured = _capture_llm(monkeypatch, ["题1"])

    synthesize_questions([], "简历", "后端开发", ["第1轮: Java内存模型 2 题", "第2轮: Spring 事务 3 题"])

    human = captured["messages"][1].content
    assert "- 第1轮: Java内存模型 2 题" in human    # 逐条带前缀进 prompt
    assert "- 第2轮: Spring 事务 3 题" in human


def test_questions_passthrough_and_purpose(monkeypatch):
    expected = ["题目A", "题目B", "题目C"]
    captured = _capture_llm(monkeypatch, expected)

    result = synthesize_questions(
        [{"text": "kb", "doc_id": "d", "chunk_index": 0, "score": 0.5}],
        "简历", "后端开发", [],
    )

    assert result == expected                      # 透传，不改写不排序
    assert captured["purpose"] is Purpose.GENERATE # 出题必须直连 strong（ADR-009）
