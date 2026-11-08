"""简历解析评分单测（mock LLM 与下载，不真调、零花费）。

三份说明（AGENT.md §8 硬性要求）：

1. 测什么 —— 落到代码行
   被测代码：`resume.py::analyze_resume()` 的三个关口：
   - 20MB 上限：`analyze_resume` 内 len(content) 超限 → VALIDATION_ERROR（本测试
     通过把 `_fetch_resume` 桩成"直接抛"来覆盖 `_fetch_resume` 内部的上限判断，
     另由 test_oversize 用真实字节覆盖 `analyze_resume` 对超大的拒绝路径）；
   - 体检行：`if len(text.strip()) < MIN_TEXT_CHARS: raise` → 断言
     `code == VALIDATION_ERROR`（对应"疑似扫描件"分支）；
   - 评分行：`structured_call(..., purpose=Purpose.SCORE)` → 断言结果透传
     （`score == 78`、`rag_query == ["Java并发"]`）且
     `captured["purpose"] is Purpose.SCORE` —— 守住"评分不进 fast"（ADR-009）。

2. 为什么测
   三个关口都是"脏输入早失败"防线：超限防内存打爆（4C4G）、扫描件防胡编分数
   （喂给 LLM 只会得到看似合理的假分）、purpose 断言防有人顺手把 SCORE 挪进
   FAST_ALLOWED（那会让评分先打免费 4B，违反 ADR-009 且 6F 配额归因失效）。

3. 抓不到什么
   mock 抓不到真实 PDF 的复杂文本问题（双栏/表格归 7F/K16）；抓不到真实 LLM
   给 120 分的行为（Pydantic ge/le 触发解析级降级，归 3D/5D 的往返测试思路）；
   也抓不到真实网络失败（httpx 异常分支）。

Mock 位置（关键行）：patch `app.agents.question_officer.resume` 命名空间里的
`_fetch_resume` / `_pdf_to_text` / `structured_call` —— 不能 patch
`app.core.llm.structured_call` 原处（resume.py 已把名字 import 进自己命名空间）。
"""
import pytest

from app.agents.question_officer import resume as resume_mod
from app.agents.question_officer.resume import ResumeAnalysis, analyze_resume
from app.core.errors import AgentError, ErrorCode
from app.core.llm import Purpose


def test_scanned_pdf_rejected(monkeypatch):
    """文本层过短（疑似扫描件）→ VALIDATION_ERROR（对应体检行 len(text)<50）。"""
    monkeypatch.setattr(resume_mod, "_fetch_resume", lambda url: b"%PDF fake")
    monkeypatch.setattr(resume_mod, "_pdf_to_text", lambda data: "  ")
    monkeypatch.setattr(
        resume_mod, "structured_call",
        lambda **kw: pytest.fail("扫描件应在评分前失败"),
    )
    with pytest.raises(AgentError) as exc_info:
        analyze_resume("http://x/resume.pdf")
    assert exc_info.value.code is ErrorCode.VALIDATION_ERROR


def test_download_error_rejected(monkeypatch):
    """下载失败（预签名过期/网络）→ VALIDATION_ERROR 而非 500（错误契约）。"""

    def fake_fetch(url):
        raise AgentError(ErrorCode.VALIDATION_ERROR, "简历下载失败: timeout")

    monkeypatch.setattr(resume_mod, "_fetch_resume", fake_fetch)
    with pytest.raises(AgentError) as exc_info:
        analyze_resume("http://x/resume.pdf")
    assert exc_info.value.code is ErrorCode.VALIDATION_ERROR


def test_score_passthrough_and_purpose(monkeypatch):
    """正常路径：结果透传 + purpose 必须是 SCORE（评分不进 fast，ADR-009）。"""
    captured = {}

    def fake_call(schema, messages, purpose):
        captured["purpose"] = purpose
        return ResumeAnalysis(score=78, resume_suggest=["补量化数据"],
                              resume_type="后端开发", rag_query=["Java并发"])

    monkeypatch.setattr(resume_mod, "_fetch_resume", lambda url: b"%PDF fake")
    monkeypatch.setattr(resume_mod, "_pdf_to_text", lambda data: "张三 Java 后端 " * 10)
    monkeypatch.setattr(resume_mod, "structured_call", fake_call)

    result = analyze_resume("http://x/resume.pdf")
    assert result.score == 78
    assert result.resume_suggest == ["补量化数据"]
    assert result.rag_query == ["Java并发"]
    assert captured["purpose"] is Purpose.SCORE
