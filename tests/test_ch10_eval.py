from ml.evaluate import compute_metrics, confusion_2x2, export_error_cases

NAMES = ("退换货", "物流", "尺码", "其他")


def test_compute_metrics_per_class_and_macro():
    gold = [["退换货"], ["物流"], ["尺码", "退换货"], ["其他"]]
    pred = [["退换货"], ["物流", "运费"], ["尺码"], ["其他"]]
    metrics = compute_metrics(gold, pred, NAMES)
    assert metrics["labels"]["退换货"]["tp"] == 1
    assert metrics["labels"]["退换货"]["fn"] == 1
    assert metrics["labels"]["物流"]["fp"] == 0
    assert "macro_f1" in metrics and "micro_f1" in metrics
    assert metrics["exact_match"] == 0.5


def test_confusion_2x2_counts():
    assert confusion_2x2([1, 1, 0, 0], [1, 0, 1, 0]) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_export_error_cases(tmp_path):
    rows = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]
    count = export_error_cases(rows, [["退换货"], ["物流"]], [["物流"], ["物流"]],
                               tmp_path / "errors.csv", NAMES, per_class=2)
    assert count == 2
    assert (tmp_path / "errors.csv").exists()
