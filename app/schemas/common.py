"""跨接口通用契约：统一错误信封。

基准：docs/server/03  错误响应 + contracts/error_envelope.json。
3B 的全局异常处理器将用它序列化所有错误响应。
"""
from pydantic import BaseModel

class ErrorDetail(BaseModel):
    code: str      # 错误码
    message: str   # 人读的描述
    trace_id: str  # 与请求同 ID，Java 凭它聚合日志

class ErrorEnvelope(BaseModel):
    error: ErrorDetail