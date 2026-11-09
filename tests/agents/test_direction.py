"""方向四级归一单测（2026-11-09 定稿：简历推导 → 用户声明 → 话题归一 → 显式失败）。

三份说明（AGENT.md §8）：

1. 测什么 —— 被测代码 `direction.py::normalize_direction(resume_type,
   direction_default, user_input)` 的四级来源循环：
   - 来源内部：`_match(candidate)` 关键词行（命中即返回）→
     `_llm_normalize(candidate, prompt, source)` 的**失败分流行**
     `if exc.code is ErrorCode.STRUCTURE_PARSE_ERROR: return None` / `raise`
     （解析失败降级下一来源；服务不可用上抛——后续来源同样依赖模型）；
   - 来源之间：`default 非枚举 → logger.warning + continue`（不透传防 Qdrant
     静默空召回，降级话题归一；最终失败时 ④ 报错带原值）；
   - 终点：`raise AgentError(VALIDATION_ERROR, ...)`（全来源落空）。
   断言对应关系：
   - test_resume_type_keyword_hit：`("Java后端", …)` → `"后端开发"` 且 LLM 未被
     调用（核心词"后端"命中，零成本行）；
   - test_resume_type_llm_hit：`("区块链开发", "后端开发", …)` → LLM 归一
     `"后端开发"`，`purpose is Purpose.EVAL`；
   - test_llm_parse_error_falls_to_default：LLM 抛 STRUCTURE_PARSE_ERROR →
     落 default `"后端开发"`（5C 的"上抛"语义已演进为"降级下一来源"——来源
     变多后某一来源失败不再意味着全链无解）；
   - test_llm_timeout_raises：LLM 抛 LLM_TIMEOUT → **上抛 LLM_TIMEOUT**
     （服务挂了，③ 再试无意义，提前止损）；
   - test_default_illegal_falls_to_topic：`(None, "全栈", "出vue题")` →
     `"前端开发"`（default 非法不透传，话题归一救回）；
   - test_all_sources_exhausted_raises：`(None, "全栈", "帮我出题")` →
     `VALIDATION_ERROR` 且报错含 default 原值（Java 据此提示选方向）；
   - test_no_resume_default_passthrough：`(None, "前端开发", "帮我出题")` →
     `"前端开发"`（透传，LLM 未被调用）；
   - test_no_default_no_topic_raises：`(None, None, "随便")` →
     `VALIDATION_ERROR`。
2. 为什么测
   归一结果直接进 Qdrant keyword 过滤：非法值 → 静默空召回；硬兜底 → 错方向。
   四级语义链的三类退化路径（来源间降级 / 服务异常上抛 / 全失败报错）各锁一条，
   防未来改动时退化成"静默错方向"（2026-11-08 用户推翻 _FALLBACK 的回归测试）。
3. 抓不到什么
   mock 抓不到真实 LLM 的归一质量（归 5F-a 评测）；关键词表覆盖盲区归人工维护；
   抓不到 Qdrant 真实过滤（4B 冒烟覆盖）。

Mock 位置：patch `app.agents.question_officer.direction` 命名空间的
`structured_call`（不能 patch `app.core.llm` 原处）。
"""
import pytest

from app.agents.question_officer import direction as direction_mod
from app.agents.question_officer.direction import (
    DIRECTIONS,
    DirectionOutput,
    normalize_direction,
)
from app.core.errors import AgentError, ErrorCode
from app.core.llm import Purpose


def _forbid_llm(monkeypatch):
    """关键词命中路径不应进 LLM——进了就 fail（零成本行被绕过）。"""
    monkeypatch.setattr(
        direction_mod, "structured_call",
        lambda **kw: pytest.fail("该来源关键词已命中，不应触发 LLM 归一"),
    )


def test_resume_type_keyword_hit(monkeypatch):
    _forbid_llm(monkeypatch)
    assert normalize_direction("Java后端", "前端开发", "随便聊聊") == "后端开发"


def test_resume_type_llm_hit(monkeypatch):
    captured = {}

    def fake_call(schema, messages, purpose):
        captured["purpose"] = purpose
        return DirectionOutput(direction="后端开发")

    monkeypatch.setattr(direction_mod, "structured_call", fake_call)
    assert normalize_direction("区块链开发", "后端开发", "帮我出题") == "后端开发"
    assert captured["purpose"] is Purpose.EVAL


def test_llm_parse_error_falls_to_default(monkeypatch):
    def fake_parse_fail(schema, messages, purpose):
        raise AgentError(ErrorCode.STRUCTURE_PARSE_ERROR, "模型输出自由文本")

    monkeypatch.setattr(direction_mod, "structured_call", fake_parse_fail)
    # resume_type 的 LLM 解析失败 → 降级 default（合法值透传）
    assert normalize_direction("区块链开发", "后端开发", "帮我出题") == "后端开发"


def test_llm_timeout_raises(monkeypatch):
    def fake_timeout(schema, messages, purpose):
        raise AgentError(ErrorCode.LLM_TIMEOUT, "[deepseek] 上游模型超时")

    monkeypatch.setattr(direction_mod, "structured_call", fake_timeout)
    with pytest.raises(AgentError) as exc_info:
        normalize_direction("区块链开发", "后端开发", "帮我出题")
    assert exc_info.value.code is ErrorCode.LLM_TIMEOUT


def test_default_illegal_falls_to_topic(monkeypatch):
    # default "全栈" 关键词 miss；default 不走 LLM（声明值只透传）→ 落 user_input
    # 话题关键词 "vue" 命中 → "前端开发"（default 非法被救回）
    assert normalize_direction(None, "全栈", "帮我出些 Vue 组件的题") == "前端开发"


def test_all_sources_exhausted_raises(monkeypatch):
    def fake_none(schema, messages, purpose):
        return DirectionOutput(direction="NONE")   # LLM 判话术无方向

    monkeypatch.setattr(direction_mod, "structured_call", fake_none)
    with pytest.raises(AgentError) as exc_info:
        normalize_direction(None, "全栈", "帮我出题")
    assert exc_info.value.code is ErrorCode.VALIDATION_ERROR
    assert "全栈" in str(exc_info.value)            # 报错带 default 原值


def test_no_resume_default_passthrough(monkeypatch):
    _forbid_llm(monkeypatch)
    assert normalize_direction(None, "前端开发", "帮我出题") == "前端开发"


def test_no_default_no_topic_raises(monkeypatch):
    def fake_none(schema, messages, purpose):
        return DirectionOutput(direction="NONE")

    monkeypatch.setattr(direction_mod, "structured_call", fake_none)
    with pytest.raises(AgentError) as exc_info:
        normalize_direction(None, None, "随便")
    assert exc_info.value.code is ErrorCode.VALIDATION_ERROR
