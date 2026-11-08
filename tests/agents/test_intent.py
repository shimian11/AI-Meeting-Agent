"""意图路由单测（mock LLM，不真调、零花费、毫秒级）。

三份说明（AGENT.md §8 硬性要求）：

1. 测什么 —— 落到代码行
   被测代码：`intent.py::classify_intent()` 中的
   `structured_call(schema=IntentOutput, messages=[...], purpose=Purpose.EVAL)`。
   断言对应关系：
   - `test_generate_passthrough`：mock 返回 GENERATE → 断言 `result.intent == "GENERATE"`，
     证明 classify_intent 把 schema 实例**原样返回**（不做二次加工）；
   - `test_smalltalk_passthrough`：同上，证明两个枚举分支都透传；
   - `test_parse_error_propagates`：mock 抛
     `AgentError(ErrorCode.STRUCTURE_PARSE_ERROR)` → 断言 `pytest.raises(AgentError)`
     且 `exc.code == STRUCTURE_PARSE_ERROR` —— 对应"解析失败禁止兜底成
     SMALL_TALK"（ADR-009）。若有人把异常吞掉改成
     `return IntentOutput(intent="SMALL_TALK")`，本条必失败。

2. 为什么测
   意图路由是黄金路径第一岔口：误判 GENERATE 会烧一次简历解析+出题（真金白银），
   误判 SMALL_TALK 直接断 GP1；"吞异常装闲聊"是 ADR-009 点名的历史事故模式。

3. 抓不到什么
   mock 抓不到真实模型的边界话术行为（"你好呀帮我出题"归 5F-a 评测集）；
   也抓不到 prompt 措辞退化（归评测集 + 人工评审）。

Mock 位置（关键行）：patch 的是 `app.agents.question_officer.intent` 命名空间里的
`structured_call` —— 若错 patch `app.core.llm.structured_call` 则不生效
（intent.py 已经把名字 import 进了自己的命名空间）。
"""
import pytest

from app.agents.question_officer import intent as intent_mod
from app.agents.question_officer.intent import IntentOutput, classify_intent
from app.core.errors import AgentError, ErrorCode


def _fake_ok(return_intent: str):
    """合法路径的 fake：按参数原样构造 schema 实例（模拟 structured_call 成功）。"""

    def fake(schema, messages, purpose):
        assert schema is IntentOutput          # schema 必须是意图枚举模型，传错要炸
        return IntentOutput(intent=return_intent)

    return fake


def test_generate_passthrough(monkeypatch):
    monkeypatch.setattr(intent_mod, "structured_call", _fake_ok("GENERATE"))
    result = classify_intent("帮我出10道Java并发题")
    assert isinstance(result, IntentOutput)
    assert result.intent == "GENERATE"


def test_smalltalk_passthrough(monkeypatch):
    monkeypatch.setattr(intent_mod, "structured_call", _fake_ok("SMALL_TALK"))
    result = classify_intent("今天天气不错啊")
    assert result.intent == "SMALL_TALK"


def test_parse_error_propagates(monkeypatch):
    """解析失败必须原样抛出，禁止兜底成 SMALL_TALK（ADR-009）。"""

    def fake_fail(schema, messages, purpose):
        raise AgentError(ErrorCode.STRUCTURE_PARSE_ERROR, "模型输出了自由文本")

    monkeypatch.setattr(intent_mod, "structured_call", fake_fail)

    with pytest.raises(AgentError) as exc_info:
        classify_intent("随便说点什么")

    assert exc_info.value.code is ErrorCode.STRUCTURE_PARSE_ERROR
