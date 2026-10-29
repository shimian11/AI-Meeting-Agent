"""
全局配置：加载环境变量。
读取顺序：环境变量>.env文件
"""

from anyio.functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="BANMA_", extra="ignore")

    env: str = "local"                  # local | test | prod
    log_level: str = "INFO"             # DEBUG | INFO | WARNING | ERROR

    agent_token: str = "change-me"      # Java与Python 内部鉴权
    llm_api_key: str = ""
    llm_base_url: str = "https://api.siliconflow.cn/v1"

    trace_full: bool = False           # True 才在 span 里记 prompt 全文（生产必须 False）
    trace_sample: float = 0.1          # 成功请求采样率 0~1 成功则采样10%，失败全采样


@lru_cache
def get_settings() -> Settings:
    return Settings()