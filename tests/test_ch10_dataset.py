import pytest

torch = pytest.importorskip("torch")

from ml.dataset import MultiLabelDataset  # noqa: E402
from ml.taxonomy import load_taxonomy  # noqa: E402


class FakeTokenizer:
    def __call__(self, text, max_length, truncation, padding):
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


def test_dataset_returns_tensors_and_multi_hot():
    tax = load_taxonomy()
    ds = MultiLabelDataset(
        [{"text": "买大了想退", "labels": ["尺码", "退换货"]}], FakeTokenizer(), tax,
    )
    item = ds[0]
    assert item["input_ids"].dtype == torch.long
    assert item["labels"].shape == (17,)
    assert item["labels"][tax.index("尺码")] == 1.0
    assert item["labels"].sum() == 2.0
