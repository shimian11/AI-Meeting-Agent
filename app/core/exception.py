"""
全局异常处理器：把所有错误统一成契约信封。
信封：{"error": {"code": str, "message": str, "trace_id": str}}
"""
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse
from loguru import logger

from fastapi import Request, FastAPI

from app.core.error import ErrorCode, AgentError, HTTP_STATUS
from app.core.trace import get_trace_id, new_trace_id


def build_envelope(status: int, code :ErrorCode, message :str) -> JSONResponse:
    """创建统一的异常信封结构"""
    trace_id = get_trace_id() or new_trace_id()
    body = {"error": {"code": code.value, "message": message, "trace_id": trace_id}}
    return JSONResponse(status_code=status, content=body, headers={"X-Trace-Id": trace_id})

async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
    logger.warning(f"业务异常 code={exc.code.value} path={request.url.path} msg={exc.message}")
    return build_envelope(exc.http_status, exc.code, exc.message)

async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """覆盖 FastAPI 默认的 422 {"detail": [...]} —— 那个格式违反契约。"""
    errors = exc.errors()
    if errors:
        first = errors[0]
        # loc 形如 ("body", "count")，去掉 body 拼成 "count: ..." 便于定位
        field_path = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        message = f"{field_path}: {first.get('msg', '参数校验失败')}" if field_path else str(first.get("msg", "参数校验失败"))
    else:
        message = "参数校验失败"
    logger.warning(f"契约校验失败 path={request.url.path} {message}")
    return build_envelope(HTTP_STATUS[ErrorCode.VALIDATION_ERROR], ErrorCode.VALIDATION_ERROR, message)

async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    框架级 HTTP 错误（404/405…）也要套信封。
    注意：FastAPI 的 HTTPException 继承自 Starlette 的，注册这一个即可全兜住。
    契约只有 7 个码，无"路径不存在"，故语义归 INTERNAL_ERROR，仅保留原始状态码。
    """
    logger.warning(f"HTTP异常 status={exc.status_code} path={request.url.path}")
    return build_envelope(exc.status_code, ErrorCode.INTERNAL_ERROR, str(exc.detail))

async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底：未分类异常 → INTERNAL_ERROR。必须打完整堆栈，否则线上无从排障。"""
    logger.exception(f"未处理异常 path={request.url.path}")
    return build_envelope(HTTP_STATUS[ErrorCode.INTERNAL_ERROR], ErrorCode.INTERNAL_ERROR, "服务内部错误")

def register_exception_handlers(app: FastAPI) -> None:
    """集中注册，避免散落在 main.py 里形成多处入口。"""
    app.add_exception_handler(AgentError, agent_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
