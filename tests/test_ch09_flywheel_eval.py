"""飞轮评估：每轮记账、同集才比、Faithfulness 劣化第一时间警告。"""

import importlib.util
import json
from datetime import datetime
from pathlib import Path

from app.services import eval_trend_service

_spec = importlib.util.spec_from_file_location(
    "flywheel_eval",
    Path(__file__).resolve().parent.parent / "eval" / "flywheel_eval.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _result(*, faithful=90, faithful_total=100, coverage=0.9, recall=0.8,
            refusal_correct=57, refusal_total=60, refused_answerable=3,
            answerable_total=240):
    return {
        "samples": 300,
        "collection_count": 1234,
        "degraded": 0,
        "started_at": "2026-10-27T09:00:00",
        "answers": {
            "answered": 300,
            "coverage": coverage,
            "faithful": faithful,
            "faithful_total": faithful_total,
            "refusal_correct": refusal_correct,
            "refusal_total": refusal_total,
            "refused_answerable": refused_answerable,
            "answerable_total": answerable_total,
        },
        "strategies": {"hybrid_rerank": {"recall@5": recall, "mrr@10": 0.6}},
    }


def _round(round_id, metrics, *, hash_="abc123"):
    return {
        "round_id": round_id, "eval_set": "ch04", "eval_set_hash": hash_,
        "metrics": metrics,
    }


def test_metrics_from_result_reads_the_honest_denominators():
    got = mod.metrics_from_result(_result())
    assert got["faithfulness"] == 0.9
    assert got["coverage"] == 0.9
    assert got["refusal_accuracy"] == 0.95
    assert got["misblocked_rate"] == 0.0125
    assert got["recall_at_5"] == 0.8


def test_metrics_from_result_retrieval_only_leaves_answer_metrics_null():
    got = mod.metrics_from_result({"strategies": {"hybrid_rerank": {"recall@5": 0.7}}})
    assert got["faithfulness"] is None
    assert got["refusal_accuracy"] is None
    assert got["recall_at_5"] == 0.7


def test_first_round_is_the_baseline():
    got = mod.compare(None, {"eval_set_hash": "abc", "metrics": {}})
    assert got["status"] == "baseline"
    assert got["deltas"] == {}


def test_same_set_rounds_compare_and_flag_faithfulness_drop():
    previous = _round("r1", {"faithfulness": 0.95, "coverage": 0.95})
    current = {"eval_set_hash": "abc123", "metrics": {"faithfulness": 0.90, "coverage": 0.95}}
    got = mod.compare(previous, current)
    assert got["status"] == "comparable"
    assert got["deltas"]["faithfulness"] == -0.05
    assert got["deltas"]["coverage"] == 0.0
    assert len(got["warnings"]) == 1
    assert "Faithfulness" in got["warnings"][0]
    assert "脏知识" in got["warnings"][0]


def test_small_wobble_does_not_warn():
    previous = _round("r1", {"faithfulness": 0.95})
    current = {"eval_set_hash": "abc123", "metrics": {"faithfulness": 0.93}}
    assert mod.compare(previous, current)["warnings"] == []


def test_changed_eval_set_is_incomparable_and_resets_baseline():
    previous = _round("r1", {"faithfulness": 0.95}, hash_="old")
    current = {"eval_set_hash": "new", "metrics": {"faithfulness": 0.5}}
    got = mod.compare(previous, current)
    assert got["status"] == "incomparable"
    assert "评估集" in got["reason"]
    assert got["deltas"] == {}


def test_run_appends_a_round_and_compares_with_the_same_set(tmp_path):
    rounds_path = tmp_path / "rounds.jsonl"
    samples = tmp_path / "samples.jsonl"
    samples.write_text("{}\n", encoding="utf-8")

    first = mod.run(
        trigger="手动", runner=lambda run_id, mode: _result(faithful=95),
        samples_path=samples, rounds_path=rounds_path,
        now=datetime(2026, 10, 27, 9, 0, 0),
    )
    second = mod.run(
        trigger="定时", runner=lambda run_id, mode: _result(faithful=88),
        samples_path=samples, rounds_path=rounds_path,
        now=datetime(2026, 10, 28, 9, 0, 0),
    )

    assert first["comparison"]["status"] == "baseline"
    assert first["trigger"] == "手动"
    assert second["comparison"]["status"] == "comparable"
    assert second["comparison"]["previous_round_id"] == first["round_id"]
    assert second["comparison"]["warnings"]
    lines = rounds_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["trigger"] for line in lines] == ["手动", "定时"]


def test_run_rejects_unknown_trigger(tmp_path):
    try:
        mod.run(trigger="按需", runner=lambda *_: _result(), rounds_path=tmp_path / "r.jsonl")
    except ValueError as e:
        assert "按需" in str(e)
    else:
        raise AssertionError("未知触发方式应当报错")


def test_read_rounds_skips_broken_lines(tmp_path):
    path = tmp_path / "rounds.jsonl"
    path.write_text('{"round_id": "r1"}\n{not json}\n\n', encoding="utf-8")
    assert [row["round_id"] for row in mod.read_rounds(path)] == ["r1"]


def test_grouped_keeps_eval_sets_on_their_own_rows():
    rounds = [
        _round("a1", {}, hash_="h1"),
        _round("b1", {}, hash_="h2"),
        _round("a2", {}, hash_="h1"),
    ]
    sets = eval_trend_service.grouped(rounds)
    by_hash = {item["eval_set_hash"]: [row["round_id"] for row in item["rounds"]] for item in sets}
    assert by_hash == {"h1": ["a1", "a2"], "h2": ["b1"]}


def test_overview_carries_the_latest_warnings(tmp_path):
    path = tmp_path / "rounds.jsonl"
    path.write_text(
        json.dumps(_round("r1", {}), ensure_ascii=False) + "\n"
        + json.dumps({**_round("r2", {}), "comparison": {"status": "comparable", "warnings": ["x"]}}) + "\n",
        encoding="utf-8",
    )
    got = eval_trend_service.overview(path)
    assert got["warnings"] == ["x"]
    assert got["latest_round_id"] == "r2"


def test_finished_at_is_taken_after_the_run_not_at_start(monkeypatch, tmp_path):
    """趋势表按完成时间排序；用开跑时刻当完成时刻，长跑那一轮会排到前面去。"""
    start = datetime(2026, 10, 27, 9, 0, 0)
    end = datetime(2026, 10, 27, 9, 30, 0)
    ticks = iter([start, end])

    class Clock:
        @staticmethod
        def now():
            return next(ticks)

    monkeypatch.setattr(mod, "datetime", Clock)
    samples = tmp_path / "samples.jsonl"
    samples.write_text("{}\n", encoding="utf-8")
    got = mod.run(
        runner=lambda run_id, mode: _result(),
        samples_path=samples,
        rounds_path=tmp_path / "rounds.jsonl",
    )
    assert got["round_id"].startswith("20261027-090000")
    assert got["finished_at"] == "2026-10-27T09:30:00"
