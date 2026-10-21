# tests/test_ch06_eval_samples.py
import json
from collections import Counter
from pathlib import Path

from app.graph.routing import INTENTS


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_intent_samples_cover_all_labels():
    rows = _rows("eval/samples_intent.jsonl")
    counts = Counter(row["intent"] for row in rows)
    assert set(counts) == set(INTENTS)
    assert counts["other"] == 10


def test_expansion_samples_have_core_and_non_core():
    rows = _rows("eval/samples_ch06_expansion.jsonl")
    assert sum(1 for row in rows if row["should_expand"]) == 10
    assert sum(1 for row in rows if not row["should_expand"]) == 5


def test_multiturn_samples_are_well_formed():
    rows = _rows("eval/samples_ch06_multiturn.jsonl")
    assert len(rows) == 8
    for group in rows:
        assert group["turns"]
        for turn in group["turns"]:
            assert turn["intent"] in INTENTS
