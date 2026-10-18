"""300 题评估集的结构契约：桶数、字段、拒答题形态。

内容是否与知识库对得上由 `eval/validate_samples_ch04.py`（连库）检查；
这里只钉住不连库也能查的部分，防止后续手改 JSONL 把结构改坏。
"""

import json
import re
from collections import Counter
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "eval" / "samples_ch04.jsonl"
BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi", "D_absent")


def _load() -> list[dict]:
    with SAMPLES.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_has_300_questions_and_60_per_bucket():
    samples = _load()
    assert len(samples) == 300
    assert Counter(s["bucket"] for s in samples) == {bucket: 60 for bucket in BUCKETS}


def test_ids_are_unique_and_match_bucket_prefix():
    samples = _load()
    ids = [s["id"] for s in samples]
    assert len(set(ids)) == len(ids)
    for sample in samples:
        assert sample["id"].startswith(sample["bucket"][0]), sample["id"]
        assert sample["id"][1:].isdigit(), sample["id"]


def test_every_question_has_query_and_behavior():
    for sample in _load():
        assert sample["query"].strip(), sample["id"]
        assert sample["expected_behavior"] in ("answer", "no_commitment", "refuse"), sample["id"]
        assert isinstance(sample["should_refuse"], bool), sample["id"]


def test_answerable_questions_have_sections_and_expected_points():
    for sample in _load():
        if sample["bucket"] == "D_absent":
            continue
        ground_truth = sample["ground_truth"]
        paths = [ground_truth["section_path"]] if ground_truth.get("section_path") else list(
            ground_truth.get("section_paths") or []
        )
        assert paths, sample["id"]
        assert sample["expected_points"], sample["id"]
        assert all(point.strip() for point in sample["expected_points"]), sample["id"]
        any_by_section = ground_truth.get("must_contain_any_by_section") or {}
        assert set(any_by_section) <= set(paths), sample["id"]


def test_absent_questions_have_no_ground_truth_and_must_refuse():
    for sample in _load():
        if sample["bucket"] != "D_absent":
            continue
        assert sample["expected_behavior"] == "refuse", sample["id"]
        assert sample["should_refuse"] is True, sample["id"]
        assert sample["ground_truth"] == {}, sample["id"]
        assert sample["expected_points"] == [], sample["id"]


def test_refund_arrival_questions_are_marked_no_commitment():
    samples = {s["id"]: s for s in _load()}
    for sample_id in ("A8", "C2", "E2", "E24"):
        assert samples[sample_id]["expected_behavior"] == "no_commitment", sample_id


def test_no_commitment_questions_do_not_require_arrival_days_as_points():
    # 到账天数正是这类题不许承诺的内容，把它写进标准要点会让正确答案永远覆盖不了
    # （E24 旧写法里的「3 个工作日」就是这么错的）。
    for sample in _load():
        if sample["expected_behavior"] != "no_commitment":
            continue
        for point in sample["expected_points"]:
            assert not re.search(r"\d+\s*个?\s*工作日", point), (sample["id"], point)


def test_old_reject_bucket_name_is_gone():
    assert all(sample["bucket"] != "D_reject" for sample in _load())
