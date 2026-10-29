"""
回放工具，按时间线打印trace的所有span，替代langsmith
"""
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("用法: uv run python -m app.scripts.trace_replay <trace_id>")
        sys.exit(1)

    trace_id = sys.argv[1]

    spans = []
    for file in sorted(Path("logs").glob("trace-*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("trace_id") == trace_id:
                spans.append(record)

    if not spans:
        print(f"未找到 trace_id={trace_id} 的 span（可能未被采样）")
        sys.exit(1)

    spans.sort(key=lambda  r: r["start"])
    print(f"trace {trace_id} 共 {len(spans)} 条 span：")
    for r in spans:
        print(f"  [{r['start']}] {r['span']:<20} {r.get('method', '')} "
              f"{r.get('path', '')} -> {r.get('status', '')} ({r.get('duration_ms', '')}ms)")

if __name__ == "__main__":
    main()