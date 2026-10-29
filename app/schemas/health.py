"""健康检查响应契约。"""
from typing import Literal

from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: Literal["ok"]  # 目前只有 ok；用 Literal 锁死取值，写错启动都不用等测试
    service: str
    version: str
    env: str