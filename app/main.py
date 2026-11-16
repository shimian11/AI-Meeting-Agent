from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, knowledge, question_officer, evaluation
from app.core.config import get_settings
from app.core.exception_handlers import register_exception_handlers
from app.core.logging import setup_logging
from app.core.trace import TraceMiddleware


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    setup_logging(settings)
    import logging
    logging.getLogger("uvicorn.error").name = "uvicorn"
    logger_cycle_msg = f"banma-agent-server 启动完成 env={settings.env}"
    from loguru import logger
    logger.info(logger_cycle_msg)
    yield
    logger.info("banma-agent-server 已关闭")

app = FastAPI(
    title="banma-agent-server",
    version="0.1.0",
    lifespan=lifespan,
)

#注册 trace 中间件
app.add_middleware(TraceMiddleware)

register_exception_handlers(app)

#设置统一 /v1前缀
app.include_router(health.router, prefix="/v1")
app.include_router(knowledge.router, prefix="/v1")
app.include_router(question_officer.router, prefix="/v1")
app.include_router(evaluation.router, prefix="/v1")

@app.get("/")
def root():
    """根路由：人工确认服务活着的便利入口，无业务作用。"""
    return {"service": "banma-agent-server", "docs": "/docs"}