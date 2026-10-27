"""置信度阈值校准：分布、扫描、推荐线、在用量下的误拦/误放。"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from app.rag.confidence import ConfidenceSignals

_spec = importlib.util.spec_from_file_location(
    "confidence_calibration",
    Path(__file__).resolve().parent.parent / "eval" / "confidence_calibration.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _rec(i, score, *, should_refuse=False, bucket="A_policy", query="q", degraded=False):
    return {
        "id": i, "bucket": bucket, "query": query, "should_refuse": should_refuse,
        "score": score, "top1": score, "effective": 0.0, "margin": 0.0, "clause": 0.0,
        "degraded": degraded,
    }


def _signals(score):
    return ConfidenceSignals(
        top1=score, effective=0.0, margin=0.0, clause=0.0, score=score, evidence_count=1,
    )


def test_percentile_uses_nearest_rank():
    values = [0.1, 0.2, 0.3, 0.4]
    assert mod.percentile(values, 0.5) == 0.2
    assert mod.percentile(values, 0.95) == 0.4
    assert mod.percentile([], 0.5) is None


def test_distribution_reports_quartiles():
    got = mod.distribution([0.9, 0.1, 0.5, 0.3])
    assert got["count"] == 4
    assert got["min"] == 0.1
    assert got["median"] == 0.3
    assert got["max"] == 0.9


def test_threshold_grid_covers_005_to_095():
    points = mod.thresholds()
    assert points[0] == 0.05
    assert points[-1] == 0.95
    assert len(points) == 91


def test_scan_rates_are_fractions_of_each_group():
    scan = mod.scan([0.9, 0.3], [0.04, 0.6])
    low = scan["thresholds"].index(0.05)
    assert scan["pass_rates"][low] == 1.0
    assert scan["false_pass_rates"][low] == 0.5


def test_recommend_takes_max_youden_and_prefers_higher_on_tie():
    scan = {
        "thresholds": [0.1, 0.2, 0.3],
        "pass_rates": [1.0, 1.0, 0.0],
        "false_pass_rates": [0.0, 0.0, 0.0],
    }
    assert mod.recommend(scan) == (0.2, 1.0)


def test_recommend_separates_the_two_piles():
    scan = mod.scan([0.8, 0.7, 0.6], [0.1, 0.2])
    threshold, youden = mod.recommend(scan)
    assert 0.2 < threshold <= 0.6
    assert youden == 1.0


def test_at_threshold_names_misblocked_and_false_passed():
    records = [
        _rec("OK", 0.6),
        _rec("BLOCKED", 0.1),
        _rec("ABSENT", 0.02, should_refuse=True),
        _rec("LEAK", 0.5, should_refuse=True),
    ]
    got = mod.at_threshold(records, 0.3)
    assert got["misblocked_ids"] == ["BLOCKED"]
    assert got["false_pass_ids"] == ["LEAK"]
    assert got["pass_rate"] == 0.5
    assert got["false_pass_rate"] == 0.5


def test_summarize_bundles_distribution_scan_and_examples():
    records = [
        _rec("A", 0.8), _rec("B", 0.2), _rec("C", 0.1, should_refuse=True),
        _rec("D", 0.9, should_refuse=True),
        _rec("D2", 0.5, should_refuse=True),
    ]
    report = mod.summarize(records, in_use=0.3, generated_at="2026-10-27T00:00:00")
    assert report["sample_count"] == 5
    assert report["answerable_count"] == 2
    assert report["refuse_count"] == 3
    assert report["distribution"]["answerable"]["max"] == 0.8
    assert report["at_in_use"]["misblocked_ids"] == ["B"]
    assert report["at_in_use"]["false_pass_ids"] == ["D", "D2"]
    assert [row["id"] for row in report["misblocked_examples"]] == ["B"]
    # 误放样例从「最自信地答错」往下排
    assert [row["id"] for row in report["false_pass_examples"]] == ["D", "D2"]
    assert report["degraded"] == []


def test_degraded_samples_are_excluded_from_the_scan():
    """精排降级没有精排分，线上也不套闸门；把它们算进分布会把可答通过率压低。"""
    records = [
        _rec("A", 0.8),
        _rec("R", 0.05, should_refuse=True),
        _rec("DEG", 0.016, degraded=True),
    ]
    report = mod.summarize(records, in_use=0.3, generated_at="2026-10-27T00:00:00")
    assert report["sample_count"] == 2
    assert report["degraded_count"] == 1
    assert report["degraded"] == ["DEG"]
    assert report["distribution"]["answerable"]["count"] == 1
    assert report["at_in_use"]["pass_rate"] == 1.0


def test_collect_records_signals_and_reports_progress():
    samples = [
        {"id": "A", "bucket": "A_policy", "query": "退货", "should_refuse": False},
        {"id": "D", "bucket": "D_absent", "query": "活体", "should_refuse": True},
    ]
    seen = []

    def fake_retrieve(sample):
        result = SimpleNamespace(rerank_degraded=False)
        return result, _signals(0.75 if sample["id"] == "A" else 0.03)

    records = mod.collect(
        samples, retrieve=fake_retrieve, progress=lambda done, total: seen.append((done, total)),
    )
    assert [row["score"] for row in records] == [0.75, 0.03]
    assert seen == [(1, 2), (2, 2)]


def test_collect_limit_takes_the_head():
    samples = [{"id": str(i), "bucket": "A_policy", "query": "q", "should_refuse": False}
               for i in range(5)]
    records = mod.collect(
        samples, limit=2, retrieve=lambda sample: (SimpleNamespace(), _signals(0.5)),
    )
    assert [row["id"] for row in records] == ["0", "1"]


def test_run_writes_and_reads_the_artifact(tmp_path):
    path = tmp_path / "calibration.json"
    report = mod.run(
        samples=[{"id": "A", "bucket": "A_policy", "query": "退货", "should_refuse": False}],
        retrieve=lambda sample: (SimpleNamespace(rerank_degraded=False), _signals(0.8)),
        in_use=0.3,
        path=path,
    )
    assert path.exists()
    assert mod.read_report(path) == report
    assert json.loads(path.read_text(encoding="utf-8"))["in_use"] == 0.3


def test_read_report_missing_or_broken_returns_none(tmp_path):
    assert mod.read_report(tmp_path / "nope.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert mod.read_report(broken) is None


def test_read_report_backfills_degraded_count_for_old_artifacts(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text('{"degraded": ["D1", "D2"]}', encoding="utf-8")
    assert mod.read_report(legacy)["degraded_count"] == 2
