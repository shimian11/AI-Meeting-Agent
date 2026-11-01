"""
全局配置：加载环境变量。
读取顺序：环境变量>.env文件
"""

from anyio.functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="BANMA_", extra="ignore")

    env: str = "local"                  # local | tests | prod
    log_level: str = "INFO"             # DEBUG | INFO | WARNING | ERROR

    agent_token: str = "change-me"      # Java与Python 内部鉴权
    # ---- 硅基流动：fast 档 + embedding 共用一把 Key ----
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_model_fast: str = "Qwen/Qwen3.5-4B"
    siliconflow_model_embed: str = "BAAI/bge-m3"  # 4D 用，先占位

    # ---- DeepSeek 官方：strong 档 ----
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model_strong: str = "deepseek-flash"

    llm_timeout_s: float = 30.0  # 单次调用超时

    # fast 档总开关：硅基免费 4B 实测首响应 71~90s（排队所致，关思考只省 5s），
    # 默认关闭 → 所有调用直连 strong（DeepSeek 官方约 1.5s）。开启后才会走
    # 「fast 失败 → 降级 strong」。依据见 ADR-014。
    fast_enabled: bool = False

    trace_full: bool = False           # True 才在 span 里记 prompt 全文（生产必须 False）
    trace_sample: float = 0.1          # 成功请求采样率 0~1 成功则采样10%，失败全采样


@lru_cache
def get_settings() -> Settings:
    return Settings()