"""
get /v1/health的断言
进程内测试：TestClient直接调app对象，不发真实网络请求（类似MockMVC）
"""
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_health_returns_200_and_contract_keys():
    """契约：200 + status=ok + service/version/env 四键齐全"""
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "banma-agent-server"
    assert "version" in body
    assert "env" in body

def test_health_always_returns_trace_header():
    """契约：中间件必须回写非空 X-Trace-Id（Java 调用方依赖它串联日志）。"""
    resp = client.get("/v1/health")
    assert resp.headers.get("x-trace-id")