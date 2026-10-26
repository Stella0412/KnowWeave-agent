import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from ml.taxonomy import load_taxonomy  # noqa: E402
from ml.train import compute_pos_weight, load_training_rows  # noqa: E402


def test_pos_weight_caps_rare_classes(tmp_path):
    tax = load_taxonomy()
    rows = [
        {"text": str(i), "labels": ["物流"] if i == 0 else ["其他"]}
        for i in range(21)
    ]
    weight = compute_pos_weight(rows, tax, cap=10.0)
    assert weight.shape == (17,)
    assert torch.all(weight >= 1.0)
    assert torch.all(weight <= 10.0)
    assert weight[tax.index("物流")] == pytest.approx(10.0, abs=1e-6)


def test_load_training_rows_reads_jsonl(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text('{"text":"a","labels":["物流"]}\n', encoding="utf-8")
    val.write_text('{"text":"b","labels":["其他"]}\n', encoding="utf-8")
    tr, va = load_training_rows(train, val)
    assert tr[0]["text"] == "a" and va[0]["text"] == "b"
