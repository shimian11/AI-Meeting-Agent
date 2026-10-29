"""
统一错误信封的契约断言（基准：docs/server/contracts/error_envelope.json）。
信封结构：{"error": {"code": str, "message": str, "trace_id": str}}
"""
import json
from pathlib import Path

from pydantic import BaseModel, ValidationError


class ErrorDetail(BaseModel):
    code: str
    message: str
    trace_id: str

class ErrorEnvelope(BaseModel):
    error: ErrorDetail

# ---- 契约样例路径：文档仓是唯一事实源，测试直接读文件 ----
CONTRACT_FILE = (
    Path(__file__).resolve().parents[4]  # AI-Meeing 仓库根
    / "docs" / "server" / "contracts" / "error_envelope.json"
)

def test_contract_sample_exists():
    """样例文件必须在——它丢了等于契约丢了。"""
    assert CONTRACT_FILE.exists(), f"契约样例缺失: {CONTRACT_FILE}"

def test_sample_passes_our_model():
    sample = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    envelope = ErrorEnvelope.model_validate(sample)
    assert envelope.error.code == "STRUCTURE_PARSE_ERROR"

def test_model_rejects_missing_code():
    """反向：缺 code 的结构必须被拒绝"""
    broken = {"error": {"message": "no code here", "trace_id": "x"}}
    try:
        ErrorEnvelope.model_validate(broken)
    except ValidationError:
        return  # 抛了才对
    raise AssertionError("缺 code 的错误信封竟然通过了校验——模型没起作用")
