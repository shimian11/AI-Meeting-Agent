"""
契约断言：7 码映射 + 信封结构 + trace_id 一致性。
设计要点：用独立 mini app 验证逐码映射，不污染生产应用的 路由表。
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.errors import AgentError, ErrorCode, HTTP_STATUS
from app.core.exception_handlers import register_exception_handlers

# ---- 1) 纯单元：7 码必须全部有状态映射，一个都不能漏 ----
def test_all_codes_have_status():
    assert set(HTTP_STATUS.keys()) == set(ErrorCode), "有错误码漏配 HTTP 状态"
    assert HTTP_STATUS[ErrorCode.AUTH_FAILED] == 401
    assert HTTP_STATUS[ErrorCode.LLM_TIMEOUT] == 504
    assert HTTP_STATUS[ErrorCode.STRUCTURE_PARSE_ERROR] == 422


# ---- 2) 逐码验证 mini app 的响应 ----
def _make_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    # 为每个错误码挂一个"必定抛它"的临时路由
    for code in ErrorCode:
        def _make_route(c=code):
            async def route():
                raise _make_error(c)
            return route
        app.add_api_route(f"/raise/{code.value}", _make_route(), methods=["GET"])

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    return app


def _make_error(code: ErrorCode) -> AgentError:
    return AgentError(code, f"测试触发 {code.value}")


def test_each_code_maps_to_contract_status():
    # raise_server_exceptions=False：否则 Exception 兜底处理器会被重新抛出（见设计原理坑 2）
    client = TestClient(_make_app(), raise_server_exceptions=False)
    for code in ErrorCode:
        resp = client.get(f"/raise/{code.value}")
        assert resp.status_code == HTTP_STATUS[code], f"{code.value} 状态码不符"
        body = resp.json()
        assert body["error"]["code"] == code.value
        assert body["error"]["message"] == f"测试触发 {code.value}"
        assert body["error"]["trace_id"]  # 非空


# ---- 3) 信封恒为三字段，多一少一都算破坏契约 ----
def test_envelope_shape_is_exact():
    client = TestClient(_make_app(), raise_server_exceptions=False)
    body = client.get("/raise/LLM_TIMEOUT").json()
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message", "trace_id"}
    assert all(isinstance(v, str) for v in body["error"].values())


# ---- 4) trace_id 与请求头同源（Java 凭它聚合日志）----
def test_trace_id_echoes_request_header():
    client = TestClient(_make_app(), raise_server_exceptions=False)
    given = "11111111222233334444555555555555"
    resp = client.get("/raise/INTERNAL_ERROR", headers={"X-Trace-Id": given})
    assert resp.json()["error"]["trace_id"] == given
    assert resp.headers["X-Trace-Id"] == given