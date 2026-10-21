"""意图分类评估：真实模型跑 70 条标注，输出准确率与混淆矩阵。

运行：uv run python eval/eval_intent.py
"""

import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.config import settings
from app.core.llm import build_structured_model
from app.graph.routing import INTENTS
from app.prompts.orchestrator import IntentOutput, intent_messages


async def main():
    model = build_structured_model(settings, IntentOutput, temperature=0.0)
    rows = [
        json.loads(line)
        for line in Path("eval/samples_intent.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    confusion = defaultdict(Counter)
    correct = 0
    parse_failures = 0
    confidences = []
    for row in rows:
        try:
            output = await model.ainvoke(intent_messages(row["query"]))
            predicted = getattr(output, "intent", "unknown")
            confidences.append(float(getattr(output, "confidence", 0.0)))
        except Exception as exc:
            parse_failures += 1
            predicted = "unknown"
            print(f"[PARSE-FAIL] {row['query']}: {exc}")
        confusion[row["intent"]][predicted] += 1
        correct += int(predicted == row["intent"])
        print(f"[{row['intent']}→{predicted}] {row['query']}")

    other_total = sum(1 for row in rows if row["intent"] == "other")
    other_correct = confusion["other"]["other"]
    print(f"\n意图准确率: {correct}/{len(rows)} = {correct / len(rows):.1%}")
    print(f"JSON 可解析率: {(len(rows) - parse_failures) / len(rows):.1%}")
    print(f"other 召回: {other_correct}/{other_total}")
    print(f"平均 confidence: {sum(confidences) / len(confidences):.3f}")
    print("\n混淆矩阵（行=标注，列=预测）:")
    for label in INTENTS:
        print(label, dict(confusion[label]))


if __name__ == "__main__":
    asyncio.run(main())
