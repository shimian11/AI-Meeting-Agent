"""方向归一化单测（三级方案，2026-11-08 定稿：default 透传 → 关键词 → LLM → 上抛）。

三份说明（AGENT.md §8）：

1. 测什么 —— 被测代码 `direction.py::normalize_direction()` 三级分支的关键行：
   - （1）透传行：`return direction_default` —— 断言 mock 的 structured_call
     **未被调用**（透传零成本，进 LLM 就是 bug）；
   - （1）校验行：`if direction_default not in DIRECTIONS: raise` —— 断言
     `AgentError` 且 `code == VALIDATION_ERROR`（枚举外 default 会让 Qdrant
     filter 命中 0 → 静默空召回，必须现在就炸）；
   - （2）关键词行：`if any(w in text for w in words)` ——
     `("Spring Boot 微服务", …)` → `"后端开发"`（技术栈词命中，LLM 未被调用）；
   - （3）LLM 行：`("区块链开发", …)` 关键词全 miss → mock 返回
     `DirectionOutput(direction="后端开发")` → 断言透传 + `purpose is Purpose.EVAL`；
   - （3）失败行：mock 抛 `AgentError(STRUCTURE_PARSE_ERROR)` → 断言**原样上抛**
     （对应 except 块的 `raise`——若有人吞掉改成 `return "后端开发"` 硬兜底，
     test_llm_failure_raises_not_fallback 必红；这正是 2026-11-08 用户修正的
     回归测试）。
2. 为什么测
   归一结果直接进 Qdrant keyword 过滤：非法值 → 静默空召回；硬兜底 → 错方向
   （前端用户收到后端题）。三级各自都有"静默出错"的退化路径，每级都要测试锁死。
3. 抓不到什么
   mock 抓不到真实 LLM 的归一质量（"区块链→后端"是否合理归 5F-a 评测）；
   抓不到关键词表的覆盖盲区演化（新框架词出现，归人工维护 + 评测）；
   抓不到有简历时 default 的非法值（该分支不消费 default，是有意收窄校验面）。

Mock 位置：patch `app.agents.question_officer.direction` 命名空间的
`structured_call`（不能 patch `app.core.llm` 原处）。
"""
import pytest

from app.agents.question_officer import direction as direction_mod
from app.agents.question_officer.direction import DirectionOutput, normalize_direction
from app.core.errors import AgentError, ErrorCode
from app.core.llm import Purpose


def _forbid_llm(monkeypatch):
    """透传/关键词命中路径不应进 LLM——进了就 fail。"""
    monkeypatch.setattr(
        direction_mod, "structured_call",
        lambda **kw: pytest.fail("该路径不应触发 LLM 归一"),
    )


def test_no_resume_passthrough_default(monkeypatch):
    """（1）无简历 → default 原样透传，零成本不进 LLM。"""
    _forbid_llm(monkeypatch)
    assert normalize_direction(None, "前端开发") == "前端开发"
    assert normalize_direction("   ", "大数据") == "大数据"     # 空白串同空


def test_no_resume_illegal_default_rejected(monkeypatch):
    """（1）default 不在四枚举 → VALIDATION_ERROR（防 Qdrant 静默空召回）。"""
    _forbid_llm(monkeypatch)
    with pytest.raises(AgentError) as exc_info:
        normalize_direction(None, "全栈")
    assert exc_info.value.code is ErrorCode.VALIDATION_ERROR


def test_keyword_hit_zero_cost(monkeypatch):
    """（2）技术栈词命中 → 零成本归一，不进 LLM。"""
    _forbid_llm(monkeypatch)
    assert normalize_direction("Spring Boot 微服务开发", "前端开发") == "后端开发"


def test_llm_normalization(monkeypatch):
    """（3）关键词全 miss → LLM 语义归一；purpose 必须是 EVAL（温度 0.2）。"""
    captured = {}

    def fake_call(schema, messages, purpose):
        captured["purpose"] = purpose
        return DirectionOutput(direction="后端开发")

    monkeypatch.setattr(direction_mod, "structured_call", fake_call)
    result = normalize_direction("区块链开发", "后端开发")

    assert result == "后端开发"
    assert captured["purpose"] is Purpose.EVAL


def test_llm_failure_raises_not_fallback(monkeypatch):
    """（3）LLM 失败 → 原样上抛；吞掉换成 return '后端开发' = 硬兜底复发，必红。"""

    def fake_fail(schema, messages, purpose):
        raise AgentError(ErrorCode.STRUCTURE_PARSE_ERROR, "两次结构化都给不出合法枚举")

    monkeypatch.setattr(direction_mod, "structured_call", fake_fail)
    with pytest.raises(AgentError) as exc_info:
        normalize_direction("区块链开发", "后端开发")
    assert exc_info.value.code is ErrorCode.STRUCTURE_PARSE_ERROR
