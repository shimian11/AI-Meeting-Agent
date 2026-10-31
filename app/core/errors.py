"""
统一错误码与业务异常
"""
from enum import Enum


class ErrorCode(str, Enum):
    """对外 7 个错误码；value 即契约里的 code 字段，禁止随意改名（Java 侧按值分支）。"""

    AUTH_FAILED = "AUTH_FAILED"                  # 401 内部 token 无效
    VALIDATION_ERROR = "VALIDATION_ERROR"        # 422 契约字段校验失败
    LLM_TIMEOUT = "LLM_TIMEOUT"                  # 504 上游大模型超时
    LLM_ERROR = "LLM_ERROR"                      # 502 上游返回异常
    STRUCTURE_PARSE_ERROR = "STRUCTURE_PARSE_ERROR"  # 422 结构化输出重试后仍失败
    QDRANT_UNAVAILABLE = "QDRANT_UNAVAILABLE"    # 503 向量库不可用
    INTERNAL_ERROR = "INTERNAL_ERROR"            # 500 未分类异常

# 错误码 → HTTP 状态
# 关键设计：映射放这里而不挂到枚举上——SSE 流式中途失败没有 HTTP 状态码，但仍需要 code。
HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.AUTH_FAILED: 401,
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.LLM_TIMEOUT: 504,
    ErrorCode.LLM_ERROR: 502,
    ErrorCode.STRUCTURE_PARSE_ERROR: 422,
    ErrorCode.QDRANT_UNAVAILABLE: 503,
    ErrorCode.INTERNAL_ERROR: 500,
}

class AgentError(Exception):
    """业务异常基类。
        业务代码只负责 raise，HTTP 状态与响应体由全局处理器统一生成
    """
    def __init__(self, code: ErrorCode, message: str):
        if not isinstance(code, ErrorCode):
            code = ErrorCode(code)
        self.code = code
        self.message = message
        super().__init__(self.message)

    @property
    def http_status(self) -> int:
        """按契约取该错误码对应的 HTTP 状态。"""
        return HTTP_STATUS[self.code]