from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")

from ml.export_onnx import check_parity  # noqa: E402

LATEST = Path("data/ch10/model/latest.txt")
MODEL_DIR = (
    Path("data/ch10/model") / LATEST.read_text(encoding="utf-8").strip() / "best"
    if LATEST.exists() else Path("data/ch10/model/missing")
)


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="未完成训练")
def test_onnx_parity_with_torch():
    onnx_path = Path("data/ch10/model/current/model.onnx")
    samples = ["买大了想退", "物流到哪了", "发票怎么开"]
    assert check_parity(MODEL_DIR, onnx_path, samples) < 1e-3
