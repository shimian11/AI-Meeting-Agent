"""真调冒烟：分别验证两家 provider 是否可用。
用法：uv run python -m app.scripts.smoke_llm
"""
import time

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.llm import Purpose, chat_text, structured_call, strip_code_fence


class SmokeOut(BaseModel):
    ok: bool
    echo: str


def main() -> None:
    s = get_settings()
    print(f"fast  = {s.siliconflow_base_url} / {s.siliconflow_model_fast}")
    print(f"strong= {s.deepseek_base_url} / {s.deepseek_model_strong}")
    print(f"keys  = siliconflow:{bool(s.siliconflow_api_key)} deepseek:{bool(s.deepseek_api_key)}")
    print(f"fast_enabled = {s.fast_enabled}   # False = 全部直连 strong（ADR-014）")
    print(f"围栏剥离: {strip_code_fence('```json\n{\"a\":1}\n```')}")

    msgs = [
        SystemMessage(content="你是测试桩，必须返回合法 JSON。"),
        HumanMessage(content='返回 {"ok": true, "echo": "hello"}'),
    ]

    def timed(label: str, fn) -> None:
        t0 = time.time()
        print(f"--- {label} ---")
        try:
            print(f"    {fn()}    耗时 {time.time() - t0:.2f}s")
        except Exception as exc:
            print(f"    [失败] {type(exc).__name__}: {str(exc)[:200]}    耗时 {time.time() - t0:.2f}s")

    # 1) GENERATE 不在 FAST_ALLOWED 内 —— 永远走 strong（DeepSeek 官方）
    timed("strong / deepseek（Purpose.GENERATE 强制 strong）",
          lambda: structured_call(SmokeOut, msgs, Purpose.GENERATE))

    # 2) EVAL 在 FAST_ALLOWED 内 —— 实际档位取决于 fast_enabled
    timed(f"EVAL 路径（fast_enabled={s.fast_enabled}）",
          lambda: structured_call(SmokeOut, msgs, Purpose.EVAL))

    # 3) 纯文本
    timed("chat_text（纯文本）",
          lambda: chat_text([HumanMessage(content="用一个词回答：你好")]))


if __name__ == "__main__":
    main()