"""
通用依赖：内部令牌鉴权（AUTH_FAILED 的唯一产生点）:
Authorization: Bearer ${BANMA_AGENT_TOKEN}
"""
from fastapi import Header

from app.core.config import get_settings
from app.core.error import AgentError, ErrorCode


def require_token(authorization: str | None = Header(default=None)) -> None:
    """校验内部令牌，失败抛 AUTH_FAILED
        用法：router = APIRouter(dependencies=[Depends(require_token)])
     """
    expected = f"Bearer {get_settings().agent_token}"
    if authorization != expected:
        raise AgentError(ErrorCode.AUTH_FAILED, "internal token invalid")