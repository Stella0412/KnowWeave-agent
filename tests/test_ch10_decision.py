from ml.decision import decide, tune_thresholds

NAMES = ("退换货", "物流", "尺码", "其他")
THRESHOLDS = (0.5, 0.5, 0.5, 0.5)


def test_decide_keeps_all_above_threshold():
    assert decide([0.9, 0.1, 0.8, 0.0], THRESHOLDS, NAMES) == ["退换货", "尺码"]


def test_decide_falls_back_to_argmax():
    assert decide([0.1, 0.4, 0.2, 0.3], THRESHOLDS, NAMES) == ["物流"]


def test_other_can_coexist_with_a_topic():
    assert decide([0.9, 0.0, 0.0, 0.8], THRESHOLDS, NAMES) == ["退换货", "其他"]
    assert decide([0.1, 0.1, 0.1, 0.9], THRESHOLDS, NAMES) == ["其他"]


def test_tune_thresholds_improves_f1():
    scores = [[0.9, 0.1], [0.6, 0.2], [0.2, 0.85], [0.1, 0.3]]
    gold = [[1, 0], [1, 0], [0, 1], [0, 0]]
    tuned = tune_thresholds(scores, gold, ("a", "b"))
    assert tuned["a"] >= 0.5
    assert set(tuned) == {"a", "b"}
