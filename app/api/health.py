"""
健康检查接口:Get /v1/health
Docker healthcheck 与 Java 侧探活使用，要求 2 秒内响应。
"""
from fastapi import APIRouter, Depends

#路由分组器
router = APIRouter()

def get_service_info() -> dict:
    from app.core.config import get_settings
    s = get_settings()
    return {
        "service": "banma-agent-server",
        "version": "0.1.0",
        "env": s.env,
    }

@router.get("/health")
def health(info: dict = Depends(get_service_info)) -> dict:
    """健康检查端点。
    info 参数由 Depends(get_service_info) 注入（≈ @Autowired 注入一个 Bean）。
    返回 dict 时 FastAPI 自动序列化为 JSON。
    """
    return {"status": "ok", **info}