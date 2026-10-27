"""证据置信度：四信号加权（权重是常量），校准只动阈值。"""

import pytest

from app.rag.confidence import (
    EFFECTIVE_COUNT_CAP,
    EFFECTIVE_SCORE_FLOOR,
    MARGIN_CAP,
    WEIGHTS,
    clause_hit,
    effective_count,
    margin_score,
    signals,
    terms,
)
from app.rag.evidence import Evidence


def _ev(i, score, question="问题", section="文档 > 节"):
    return Evidence(
        id=i, question=question, answer="答案", section_path=section,
        content_type="policy", row_category="", category="文档", score=score,
    )


def test_weights_are_the_documented_constants():
    assert WEIGHTS == {"top1": 0.5, "effective": 0.2, "margin": 0.2, "clause": 0.1}
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_terms_keep_model_words_and_chinese_bigrams():
    got = terms("MH-LP100 的废砂盒多久倒一次")
    assert {"mh", "lp100", "废砂", "砂盒"} <= got
    assert "的" not in got


def test_effective_count_floors_and_caps():
    assert effective_count([]) == 0.0
    assert effective_count([0.05, 0.02]) == 0.0
    assert effective_count([0.2]) == pytest.approx(1 / EFFECTIVE_COUNT_CAP)
    assert effective_count([0.9, 0.8, 0.7, 0.6]) == 1.0
    assert EFFECTIVE_SCORE_FLOOR == 0.1


def test_margin_uses_absolute_gap_and_caps():
    assert margin_score([0.5]) == 0.0          # 只有一条，没有分差可言
    assert margin_score([0.30, 0.29]) == pytest.approx(0.05)
    assert margin_score([0.9, 0.1]) == 1.0
    assert MARGIN_CAP == 0.2


def test_clause_hit_matches_library_phrasing_only():
    items = [_ev(1, 0.9, question="退货政策是什么样的")]
    assert clause_hit("退货政策", items) == 1.0
    # 用户措辞与库里问法离得远 —— 这就是「可答被误拦」的成因
    assert clause_hit("能扔洗衣机里洗吗", [_ev(1, 0.9, question="猫窝清洗保养说明")]) == 0.0


def test_signals_combine_in_order_and_clamp():
    items = [
        _ev(1, 0.9, question="退货政策是什么样的"),
        _ev(2, 0.3, question="怎么申请退款"),
    ]
    got = signals(items, "退货政策")
    assert got.top1 == pytest.approx(0.9)
    assert got.effective == pytest.approx(2 / EFFECTIVE_COUNT_CAP)
    assert got.margin == 1.0
    assert got.clause > 0
    expected = 0.5 * 0.9 + 0.2 * (2 / 3) + 0.2 * 1.0 + 0.1 * got.clause
    assert got.score == pytest.approx(min(expected, 1.0))
    assert got.evidence_count == 2


def test_weak_evidence_lands_far_below_a_healthy_threshold():
    items = [
        _ev(1, 0.013, question="退货政策是什么样的"),
        _ev(2, 0.012, question="退款时效"),
    ]
    got = signals(items, "你们卖不卖活体宠物")
    assert got.score < 0.05


def test_signals_empty_items_are_zero():
    got = signals([], "退货政策")
    assert got.score == 0.0
    assert got.evidence_count == 0
