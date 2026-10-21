"""逐轮跑指代消解 + 意图，校验多轮验收。运行：uv run python eval/eval_coref.py"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.core.llm import build_structured_model
from app.prompts.orchestrator import (
    CorefOutput, IntentOutput, coref_messages, intent_messages,
)


async def main() -> int:
    coref_model = build_structured_model(settings, CorefOutput, temperature=0.0)
    intent_model = build_structured_model(settings, IntentOutput, temperature=0.0)
    rows = [json.loads(line) for line in Path("eval/samples_ch06_multiturn.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    failures = 0
    for group in rows:
        history = []
        for turn in group["turns"]:
            current = turn["user"]
            resolved_out = await coref_model.ainvoke(coref_messages(history, current))
            resolved = getattr(resolved_out, "resolved_query", current) or current
            intent_out = await intent_model.ainvoke(intent_messages(resolved, history))
            intent = getattr(intent_out, "intent", "unknown")
            missing = [word for word in turn.get("must_include", []) if word not in resolved]
            ok = intent == turn["intent"] and not missing
            print(f"[{'PASS' if ok else 'FAIL'}] {group['id']} {current!r} -> {resolved!r} / {intent}")
            if not ok:
                failures += 1
                print(f"       expected intent={turn['intent']} missing={missing}")
            history.extend([HumanMessage(content=current), AIMessage(content=resolved)])
    print(f"\n多轮验收: {'PASS' if failures == 0 else f'FAIL ({failures} turns)'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
