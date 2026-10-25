import pytest

from eval.eval_ch09_merge import cosine, load_pairs, scan_thresholds


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
