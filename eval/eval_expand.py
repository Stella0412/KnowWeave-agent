"""扩写评估：核心场景严格 3 条且唯一；非核心场景不应路由 refund_flow。"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.config import settings
from app.core.llm import build_structured_model
from app.graph.routing import ROUTE_BY_INTENT
from app.prompts.orchestrator import ExpansionOutput, expansion_messages


async def main() -> int:
    model = build_structured_model(settings, ExpansionOutput, temperature=0.0)
    rows = [json.loads(line) for line in Path("eval/samples_ch06_expansion.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    failures = 0
    for row in rows:
        if not row["should_expand"]:
            ok = ROUTE_BY_INTENT[row["intent"]] != "refund_flow"
            print(f"[{'PASS' if ok else 'FAIL'}] non-core {row['query']}")
            failures += int(not ok)
            continue
        try:
            out = await model.ainvoke(expansion_messages(row["query"], row["intent"], {}))
            queries = list(getattr(out, "queries", []) or [])
            ok = len(queries) == 3 and len(set(queries)) == 3 and all(q.strip() for q in queries)
        except Exception as exc:
            queries = []
            ok = False
            print(f"       error: {exc}")
        print(f"[{'PASS' if ok else 'FAIL'}] {row['query']} -> {queries}")
        failures += int(not ok)
    print(f"\n扩写验收: {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
