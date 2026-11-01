"""离线测试：温度分档 / 围栏剥离 / provider 绑定 / 双层降级。不访问网络。"""
import openai
import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.core import llm
from app.core.config import get_settings
from app.core.errors import AgentError, ErrorCode
from app.core.llm import FAST_ALLOWED, TEMPERATURE, Purpose, strip_code_fence


class _Demo(BaseModel):
    ok: bool

@pytest.fixture(autouse=True)
def _fake_keys(monkeypatch):
    """单元测试不依赖本地 .env：统一注入假 Key，任何机器都能跑绿。"""
    s = get_settings()
    monkeypatch.setattr(s, "siliconflow_api_key", "sk-test-sf")
    monkeypatch.setattr(s, "deepseek_api_key", "sk-test-ds")
    # 默认关闭 fast（与生产默认一致）；测降级路径的用例自行打开
    monkeypatch.setattr(s, "fast_enabled", False)
    # 降级计数器是模块级全局，跨用例会互相污染 → 每个用例从空开始
    monkeypatch.setattr(llm, "_fallback_counts", {})


class _FakeRunnable:
    """假链路：可返回固定结果或抛指定异常。"""

    def __init__(self, result=None, boom=None):
        self._result = result
        self._boom = boom

    def invoke(self, _messages):
        if self._boom:
            raise self._boom
        return self._result


class _FakeModel:
    def __init__(self, runnable):
        self._runnable = runnable

    def with_structured_output(self, _schema, **_kwargs):
        return self._runnable


def test_temperature_tiers():
    assert TEMPERATURE[Purpose.CHAT] == 0.7
    assert TEMPERATURE[Purpose.EVAL] == 0.2
    assert TEMPERATURE[Purpose.GENERATE] == 0.5


def test_generate_never_uses_fast():
    assert Purpose.GENERATE not in FAST_ALLOWED
    assert Purpose.CHAT in FAST_ALLOWED and Purpose.EVAL in FAST_ALLOWED


@pytest.mark.parametrize("raw,expected", [
    ('```json\n{"a":1}\n```', '{"a":1}'),
    ('```\n{"a":1}\n```', '{"a":1}'),
    ('{"a":1}', '{"a":1}'),
    ('  {"a":1}  ', '{"a":1}'),
    ("", ""),
])
def test_strip_code_fence(raw, expected):
    assert strip_code_fence(raw) == expected


def test_provider_binding(monkeypatch):
    """fast 必须绑硅基、strong 必须绑官方——四元组成组，不许只换 model。"""
    s = get_settings()
    monkeypatch.setattr(s, "siliconflow_api_key", "sk-sf")
    monkeypatch.setattr(s, "deepseek_api_key", "sk-ds")

    fast = llm._provider_of(llm.Tier.FAST)
    strong = llm._provider_of(llm.Tier.STRONG)

    assert fast[0] == "siliconflow" and "siliconflow" in fast[1]
    assert fast[3] == s.siliconflow_model_fast
    assert strong[0] == "deepseek" and "deepseek.com" in strong[1]
    assert strong[3] == s.deepseek_model_strong


@pytest.mark.parametrize("attr,missing", [
    ("siliconflow_api_key", "SILICONFLOW"),
    ("deepseek_api_key", "DEEPSEEK"),
])
def test_missing_key_raises(monkeypatch, attr, missing):
    s = get_settings()
    monkeypatch.setattr(s, attr, "")
    # FA/SILICONFLOW 走 FAST，DEEPSEEK 走 STRONG
    tier = llm.Tier.FAST if missing == "SILICONFLOW" else llm.Tier.STRONG
    with pytest.raises(AgentError) as ei:
        llm._provider_of(tier)
    assert ei.value.code is ErrorCode.LLM_ERROR
    assert missing in ei.value.message


@pytest.mark.parametrize("exc,expected", [
    (openai.APITimeoutError(request=None), ErrorCode.LLM_TIMEOUT),
    (openai.APIConnectionError(request=None), ErrorCode.LLM_ERROR),
])
def test_upstream_error_mapping_has_provider(exc, expected):
    err = llm._map_upstream_error(exc, "deepseek")
    assert err.code is expected
    assert err.message.startswith("[deepseek]")   # 排障要能分清是哪家


def test_exception_level_escalation(monkeypatch):
    """第一道：fast 抛超时 -> 升 strong 成功，且计数 +1。"""
    calls = []
    monkeypatch.setattr(get_settings(), "fast_enabled", True)   # 降级路径需要 fast 开着

    def fake_build(tier, purpose):
        calls.append(tier)
        if tier is llm.Tier.FAST:
            return _FakeModel(_FakeRunnable(boom=openai.APITimeoutError(request=None)))
        return _FakeModel(_FakeRunnable({"raw": object(), "parsed": _Demo(ok=True), "parsing_error": None}))

    monkeypatch.setattr(llm, "_build_model", fake_build)
    before = llm.fallback_counts().get("fast_failed_LLM_TIMEOUT", 0)

    out = llm.structured_call(_Demo, [HumanMessage(content="hi")], Purpose.EVAL)

    assert out == _Demo(ok=True)
    assert calls[0] is llm.Tier.FAST and calls[-1] is llm.Tier.STRONG
    assert llm.fallback_counts()["fast_failed_LLM_TIMEOUT"] == before + 1


def test_parse_level_escalation(monkeypatch):
    """第二道：fast 返回了但 parsed 为空 -> 用 strong 重试一次。"""
    calls = []
    monkeypatch.setattr(get_settings(), "fast_enabled", True)   # 降级路径需要 fast 开着

    def fake_build(tier, purpose):
        calls.append(tier)
        if len(calls) == 1:
            return _FakeModel(_FakeRunnable({"raw": object(), "parsed": None, "parsing_error": ValueError("坏 JSON")}))
        return _FakeModel(_FakeRunnable({"raw": object(), "parsed": _Demo(ok=True), "parsing_error": None}))

    monkeypatch.setattr(llm, "_build_model", fake_build)
    out = llm.structured_call(_Demo, [HumanMessage(content="hi")], Purpose.EVAL)

    assert out == _Demo(ok=True)
    assert len(calls) >= 2


def test_thinking_disabled_per_provider(monkeypatch):
    """两家都必须关思考，但字段名不同：deepseek 用 thinking，硅基用 enable_thinking。

    为什么必须关（ADR-014）：
      · DeepSeek 官方默认开思考，而思考模式不支持 tool_choice → 结构化输出 400
      · 硅基 Qwen3.5 默认开思考 → 白烧 token（实测 309 里 307 是思考）
      · 思考模式下 temperature 被忽略，温度分档会失效
    """
    s = get_settings()
    monkeypatch.setattr(s, "siliconflow_api_key", "sk-sf")
    monkeypatch.setattr(s, "deepseek_api_key", "sk-ds")

    strong = llm._build_model(llm.Tier.STRONG, Purpose.GENERATE)
    fast = llm._build_model(llm.Tier.FAST, Purpose.EVAL)

    assert strong.extra_body == {"thinking": {"type": "disabled"}}
    assert fast.extra_body == {"enable_thinking": False}


def test_fast_disabled_goes_straight_to_strong(monkeypatch):
    """fast 关闭时不得碰 fast 档，且不产生降级计数。"""
    monkeypatch.setattr(get_settings(), "fast_enabled", False)
    calls = []

    def fake_build(tier, purpose):
        calls.append(tier)
        return _FakeModel(
            _FakeRunnable({"raw": object(), "parsed": _Demo(ok=True), "parsing_error": None})
        )

    monkeypatch.setattr(llm, "_build_model", fake_build)
    out = llm.structured_call(_Demo, [HumanMessage(content="hi")], Purpose.EVAL)

    assert out == _Demo(ok=True)
    assert calls == [llm.Tier.STRONG]                     # 只调一次，且是 strong
    assert "fast_failed_LLM_TIMEOUT" not in llm.fallback_counts()