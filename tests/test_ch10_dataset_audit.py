import json

import pytest

from ml.dataset_audit import DatasetError, audit_dataset, load_split, write_manifest


def _write(root, split, rows):
    path = root / f"{split}.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _make_dataset(root, train, val, test):
    _write(root, "train", train)
    _write(root, "val", val)
    _write(root, "test", test)


def test_audit_counts_support_and_multi_label(tmp_path):
    _make_dataset(
        tmp_path,
        [
            {"text": "买大了想退", "labels": ["尺码", "退换货"]},
            {"text": "物流到哪了", "labels": ["物流"]},
        ],
        [{"text": "发票怎么开", "labels": ["发票"]}],
        [{"text": "能补差价吗", "labels": ["价保"]}],
    )
    manifest = audit_dataset(tmp_path)
    assert manifest["splits"]["train"]["rows"] == 2
    assert manifest["splits"]["train"]["multi_label"] == 1
    assert manifest["splits"]["test"]["labels"]["价保"] == 1
    assert manifest["totals"]["rows"] == 4
    assert manifest["taxonomy_version"] == "1.0.0"
    assert manifest["pii_hits"] == 0
    assert not any("PII" in warning for warning in manifest["warnings"])


def test_unknown_label_rejected(tmp_path):
    _make_dataset(
        tmp_path,
        [{"text": "x", "labels": ["不存在的类"]}],
        [{"text": "y", "labels": ["物流"]}],
        [{"text": "z", "labels": ["物流"]}],
    )
    with pytest.raises(DatasetError, match="未知标签"):
        audit_dataset(tmp_path)


def test_empty_labels_rejected(tmp_path):
    _make_dataset(
        tmp_path,
        [{"text": "x", "labels": []}],
        [{"text": "y", "labels": ["物流"]}],
        [{"text": "z", "labels": ["物流"]}],
    )
    with pytest.raises(DatasetError, match="labels 不能为空"):
        audit_dataset(tmp_path)


def test_other_can_coexist_with_a_topic(tmp_path):
    _make_dataset(
        tmp_path,
        [{"text": "能转人工吗，顺便问下快递到哪了", "labels": ["其他", "物流"]}],
        [{"text": "y", "labels": ["物流"]}],
        [{"text": "z", "labels": ["物流"]}],
    )
    manifest = audit_dataset(tmp_path)
    assert manifest["totals"]["other_coexist"] == 1
    assert manifest["splits"]["train"]["labels"]["其他"] == 1
    assert manifest["splits"]["train"]["labels"]["物流"] == 1


def test_cross_split_exact_duplicate_rejected(tmp_path):
    _make_dataset(
        tmp_path,
        [{"text": "同一句话", "labels": ["物流"]}],
        [{"text": "同一句话", "labels": ["物流"]}],
        [{"text": "z", "labels": ["物流"]}],
    )
    with pytest.raises(DatasetError, match="跨 split"):
        audit_dataset(tmp_path)


def test_pii_is_reported_as_warning(tmp_path):
    _make_dataset(
        tmp_path,
        [{"text": "联系我 13800138000", "labels": ["物流"]}],
        [{"text": "y", "labels": ["物流"]}],
        [{"text": "z", "labels": ["物流"]}],
    )
    manifest = audit_dataset(tmp_path)
    assert manifest["pii_hits"] == 1
    assert any("PII" in warning for warning in manifest["warnings"])


def test_write_manifest_and_load_split(tmp_path):
    _make_dataset(
        tmp_path,
        [{"text": "x", "labels": ["物流"]}],
        [{"text": "y", "labels": ["物流"]}],
        [{"text": "z", "labels": ["物流"]}],
    )
    out = write_manifest(tmp_path)
    assert out == tmp_path / "manifest.json"
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert len(saved["splits"]["train"]["sha256"]) == 64
    assert load_split(tmp_path / "train.jsonl")[0]["text"] == "x"
