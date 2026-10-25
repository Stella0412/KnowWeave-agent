import pytest

from eval.eval_ch09_merge import cosine, load_pairs, recommend_threshold, scan_thresholds


def test_cosine_identical_orthogonal_and_zero():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine([], []) == 0.0


def test_scan_thresholds_metrics():
    pairs = [
        {"a": "同", "b": "同", "label": "dup"},
        {"a": "同", "b": "异", "label": "distinct"},
    ]
    vectors = [
        ([1.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [0.0, 1.0]),
    ]
    row = scan_thresholds(pairs, vectors, [0.5])[0]
    assert row["threshold"] == 0.5
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["f1"] == 1.0


def test_load_pairs_rejects_too_few(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join('{"a":"x","b":"y","label":"dup"}' for _ in range(5)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_pairs(path)


def test_load_pairs_rejects_bad_row(tmp_path):
    path = tmp_path / "pairs.jsonl"
    lines = ['{"a":"x","b":"y","label":"dup"}'] * 49 + ['{"a":"x","b":"y","label":"weird"}']
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError):
        load_pairs(path)


def test_recommend_threshold_picks_highest_recall_at_min_precision():
    rows = [
        {"threshold": 0.90, "precision": 0.80, "recall": 0.99, "f1": 0.88},
        {"threshold": 0.93, "precision": 0.95, "recall": 0.70, "f1": 0.80},
        {"threshold": 0.96, "precision": 0.98, "recall": 0.65, "f1": 0.78},
    ]
    best = recommend_threshold(rows)
    assert best["threshold"] == 0.93
    assert best["recall"] == 0.70


def test_recommend_threshold_ties_break_to_higher_threshold():
    rows = [
        {"threshold": 0.94, "precision": 0.96, "recall": 0.80, "f1": 0.87},
        {"threshold": 0.96, "precision": 0.97, "recall": 0.80, "f1": 0.88},
    ]
    assert recommend_threshold(rows)["threshold"] == 0.96


def test_recommend_threshold_none_when_precision_too_low():
    rows = [{"threshold": 0.90, "precision": 0.80, "recall": 0.99, "f1": 0.88}]
    assert recommend_threshold(rows) is None
