"""日志装配：stdout 走 JSON（给 docker logs / 收集器），文件按天滚动、保留 14 天"""

import sys

from loguru import logger

from app.core.config import Settings

def setup_logging(settings: Settings):
    logger.remove()

    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        serialize=True,
    )

    logger.add(
        "logs/app-{time:YYYY-MM-DD}.log",
        level=settings.log_level.upper(),
        serialize=True,
        rotation="00:00",
        retention="14 days",
        encoding="utf-8",
    )

