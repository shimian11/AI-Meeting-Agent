"""
GET /v1/health 的断言

进程内测试：TestClient 直接调 app 对象，不发真实网络请求（类似 MockMvc）。
键集合从契约样例 health_resp.json 读取——改字段只改样例一处，测试自动跟随。
"""
import json
from pathlib import Path

from starlette.testclient import TestClient

from app.main import app

CONTRACT_DIR = Path(__file__).resolve().parents[4] / "docs" / "server" / "contracts"
client = TestClient(app)


def _contract_keys() -> set[str]:
    return set(json.loads((CONTRACT_DIR / "health_resp.json").read_text(encoding="utf-8")))


def test_health_returns_200_and_contract_keys():
    """契约：200 + 响应的键集合必须与 health_resp.json 完全一致（不多不少）"""
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body.keys()) == _contract_keys()


def test_health_service_name_fixed():
    """契约：service 名固定——Java 侧探活据此判断，改了会误判服务不可用"""
    body = client.get("/v1/health").json()
    assert body["service"] == "banma-agent-server"


def test_health_always_returns_trace_header():
    """契约：中间件必须回写非空 X-Trace-Id（Java 调用方依赖它串联日志）"""
    resp = client.get("/v1/health")
    assert resp.headers.get("x-trace-id")
