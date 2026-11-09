"""出题官接口契约测试（TestClient 进程内 + mock 装配层，不真调）。

三份说明（AGENT.md §8）：

1. 测什么 —— 被测代码：
   - 鉴权：`deps.py::require_token` 的 `if authorization != expected: raise AUTH_FAILED`
     （经 router 级 `dependencies=[Depends(require_token)]` 生效）——断言无 token
     返回 401 且信封 `code == "AUTH_FAILED"`（3B 的 build_envelope 覆盖 FastAPI
     默认 {"detail":…} 的回归）；
   - 装配接线：`api/question_officer.py` 的 `return run_question_officer(req)` ——
     断言带合法 token 时返回 200 且 body 含契约四键（trace_id/intent/questions/
     questions_source），证明 HTTP 层 → 装配层 → 契约模型整条接线通。
2. 为什么测
   鉴权漏挂 = 任何拿到内网地址的人都能烧 LLM 额度（成本红线）；信封格式回归 =
   Java 侧 6B 解析依赖 error_envelope.json，FastAPI 默认格式会让 Java 解析炸。
3. 抓不到什么
   TestClient 不过真实网络（header 大小写/超时归 6B 联调）；mock 层抓不到五步
   真实行为（归 test_officer.py 与 5F-a）。

Mock 位置：patch `app.api.question_officer` 命名空间的 `run_question_officer`
（端点模块已 import 进自己命名空间）。
"""
from fastapi.testclient import TestClient

from app.api import question_officer as api_mod
from app.main import app
from app.schemas.question_officer import QuestionOfficerResponse

client = TestClient(app)
_HEADERS = {"Authorization": "Bearer change-me-in-prod"}
_BODY = {
    "trace_id": "t-9", "session_id": "s-9", "user_input": "帮我出题",
    "direction_default": "后端开发", "history_digest": [],
}


def test_missing_token_returns_auth_envelope():
    resp = client.post("/v1/agents/question-officer/run", json=_BODY)
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "AUTH_FAILED"          # 信封而非 {"detail":…}
    assert "trace_id" in body["error"]


def test_run_with_token_returns_contract(monkeypatch):
    monkeypatch.setattr(
        api_mod, "run_question_officer",
        lambda req: QuestionOfficerResponse(
            trace_id=req.trace_id, intent="GENERATE", questions=["题1"],
            resume_score=None, resume_suggest=[], resume_type=None,
            questions_source="none", small_talk=None, steps_trace={},
        ),
    )
    resp = client.post("/v1/agents/question-officer/run", json=_BODY, headers=_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    for key in ("trace_id", "intent", "questions", "questions_source"):
        assert key in body
    assert body["trace_id"] == "t-9"
