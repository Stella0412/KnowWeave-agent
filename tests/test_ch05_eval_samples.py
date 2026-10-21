import json
from collections import Counter
from pathlib import Path

from app.graph.routing import INTENTS


def test_intent_samples_cover_eight_labels():
    rows = [
        json.loads(line)
        for line in Path("eval/samples_intent.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 70
    counts = Counter(row["intent"] for row in rows)
    assert set(counts) == set(INTENTS)
    assert all(counts[label] >= 10 for label in INTENTS)
    assert all(row["query"].strip() for row in rows)
