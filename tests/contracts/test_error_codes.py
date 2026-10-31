"""
契约断言：7 码映射 + 信封结构 + trace_id 一致性。
设计要点：用独立 mini app 验证逐码映射，不污染生产应用的 路由表。
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.errors import AgentError, ErrorCode, HTTP_STATUS
from app.core.exception_handlers import register_exception_handlers
from app.core.trace import TraceMiddleware


# ---- 1) 纯单元：7 码必须全部有状态映射，一个都不能漏 ----
def test_all_codes_have_status():
    assert set(HTTP_STATUS.keys()) == set(ErrorCode), "有错误码漏配 HTTP 状态"
    assert HTTP_STATUS[ErrorCode.AUTH_FAILED] == 401
    assert HTTP_STATUS[ErrorCode.LLM_TIMEOUT] == 504
    assert HTTP_STATUS[ErrorCode.STRUCTURE_PARSE_ERROR] == 422


# ---- 2) 逐码验证 mini app 的响应 ----
def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(TraceMiddleware)
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

# ---- 5) 兜底：调用方没带 X-Trace-Id 时必须自动生成 32 位 hex ----
def test_trace_id_generated_when_header_missing():
    """缺 X-Trace-Id 时自动生成，且格式是 uuid4().hex。"""
    client = TestClient(_make_app(), raise_server_exceptions=False)
    tid = client.get("/raise/LLM_TIMEOUT").json()["error"]["trace_id"]

    assert len(tid) == 32
    assert all(c in "0123456789abcdef" for c in tid)


# ---- 6) 装配回归：真实 app 必须真的装了中间件与处理器 ----
def test_real_app_404_echoes_trace_id():
    """
    防回归：main.py 若漏装 TraceMiddleware 或漏注册 handler，此测试必红。
    这是唯一覆盖"真实装配"的测试——mini app 只能证明处理器逻辑对，
    证明不了 main.py 把它们接上了。
    """
    from app.main import app as real_app   # 函数内 import：避免模块级副作用影响其他用例

    client = TestClient(real_app, raise_server_exceptions=False)
    given = "aaaaaaaabbbbbbbbccccccccdddddddd"
    resp = client.get("/v1/no-such-route", headers={"X-Trace-Id": given})

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "INTERNAL_ERROR"
    assert resp.json()["error"]["trace_id"] == given
    assert resp.headers["X-Trace-Id"] == given
