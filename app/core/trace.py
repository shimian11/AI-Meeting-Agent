"""
trace骨架
trace_id 从 X-Trace-Id 贯穿 → 每个 HTTP 请求写一条 span 到 trace-*.jsonl。
采样：成功按 trace_id 哈希确定性采样（默认 10%），失败 100% 全采。
"""
import contextvars
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from loguru import logger

current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
sampled: contextvars.ContextVar[bool] = contextvars.ContextVar("sampled", default=False)

TRACE_DIR = Path("logs")

def new_trace_id() -> str:
    return uuid.uuid4().hex

def get_trace_id() -> str:
    return current_trace_id.get()

def is_sampled(trace_id: str, rate: float) -> bool:
    """确定性采样：同一 trace_id 永远得到同一结论（哈希取模，非随机）"""
    h = int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < rate

def span_prompt_fields(prompt: str) -> dict:
    """
    prompt 脱敏：默认只记哈希+长度（3C 调 LLM 时用）。
    BANMA_TRACE_FULL=true 时才记全文 —— 本地调试开关，生产必须 False。
    """
    from app.core.config import get_settings
    s = get_settings()
    if s.trace_full:
        return {"prompt": prompt}
    return{
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16] if prompt else "",
        "prompt_len": len(prompt),
    }

def write_span(name: str, force: bool = False, **fields) -> None:
    """写一条 span 到 logs/trace-YYYY-MM-DD.jsonl。未命中采样且非失败 → 丢弃。"""
    if not(sampled.get() or force):
        return
    TRACE_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    record = {
        "trace_id": current_trace_id.get(),
        "span": name,
        "start": now.isoformat(),
        **fields,
    }
    file = TRACE_DIR / f"trace-{now:%Y-%m-%d}.jsonl"
    with file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

class TraceMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("x-trace-id") or new_trace_id()
        current_trace_id.set(trace_id)
        sampled.set(is_sampled(trace_id, get_settings().trace_sample))

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Trace-Id"] = trace_id       # 响应头回传，前端/调用方可追踪
            return response
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            # 失败强制全采（ADR-010）：4xx/5xx 的 span 一定落盘
            write_span("http.request", force=status_code >= 400,
                       method=request.method, path=request.url.path,
                       status=status_code, duration_ms=duration_ms)
            logger.bind(trace_id=trace_id).info(
                "http {method} {path} -> {status} {duration_ms}ms",
                method=request.method, path=request.url.path,
                status=status_code, duration_ms=duration_ms,
            )


from app.core.config import get_settings


