from app.rag.evidence import Evidence
from app.rag.refusal import (
    REFUSAL_TEXT,
    SOURCE_RETRIEVAL,
    SOURCE_SELF_CHECK,
    pool_record,
    retrieval_gate,
    self_check_gate,
)
from app.rag.strategies import BM25, HYBRID, HYBRID_RERANK, VECTOR


def _ev(i, score):
    return Evidence(
        id=i, question="q", answer="a", section_path="文档 > 节", content_type="policy",
        row_category="", category="文档", score=score,
    )


def test_empty_recall_refuses_for_every_strategy():
    for strategy in (VECTOR, BM25, HYBRID, HYBRID_RERANK):
        decision = retrieval_gate(strategy, [])
        assert decision.refused is True
        assert decision.source == SOURCE_RETRIEVAL
        assert "空" in decision.reason


def test_rerank_strategy_refuses_below_threshold():
    decision = retrieval_gate(
        HYBRID_RERANK, [_ev(1, 0.5)], rerank_score=0.04, rerank_min_score=0.10
    )
    assert decision.refused is True
    assert "0.04" in decision.reason and "0.10" in decision.reason


def test_rerank_strategy_passes_above_threshold():
    decision = retrieval_gate(
        HYBRID_RERANK, [_ev(1, 0.5)], rerank_score=0.65, rerank_min_score=0.10
    )
    assert decision.refused is False


def test_vector_strategy_uses_dense_threshold():
    assert retrieval_gate(VECTOR, [_ev(1, 0.4)], dense_min_score=0.55).refused is True
    assert retrieval_gate(VECTOR, [_ev(1, 0.8)], dense_min_score=0.55).refused is False


def test_bm25_strategy_has_no_threshold():
    """BM25 是负数无界分，没有稳定量纲 → 除空结果外不拒（spec §9.2）。"""
    assert retrieval_gate(BM25, [_ev(1, -9.5)]).refused is False


def test_hybrid_without_rerank_only_checks_emptiness():
    """RRF 分是名次倒数，与相似度不同量纲，不能拿来比阈值（附录 A.3）。"""
    assert retrieval_gate(HYBRID, [_ev(1, 0.016)]).refused is False


def test_rerank_degraded_does_not_apply_the_rerank_threshold():
    """Ruling R1（本章最重要的一条）：精排上游挂掉时 rerank_score 传 None，而候选分是
    hybrid_search 的 RRF 名次倒数（≈0.016~0.033）。若这时还去比 rerank_min_score=0.10，
    **每一道题都会被拒答**（答案率掉到 0），根本不是「优雅降级」。降级必须等价于纯 hybrid：
    除空结果外不拒。"""
    assert retrieval_gate(HYBRID_RERANK, [_ev(1, 0.0164), _ev(2, 0.0328)]).refused is False


def test_vector_strategy_without_threshold_argument_uses_settings(monkeypatch):
    from app.rag import refusal

    monkeypatch.setattr(refusal.settings, "retrieval_min_score", 0.9)
    assert retrieval_gate(VECTOR, [_ev(1, 0.5)]).refused is True


def test_self_check_gate_refuses_with_model_reason():
    decision = self_check_gate(False, "资料里只说了到账以支付渠道为准，没有具体天数")
    assert decision.refused is True
    assert decision.source == SOURCE_SELF_CHECK
    assert "支付渠道" in decision.reason
    assert self_check_gate(True, "资料足够").refused is False


def test_pool_record_shape_matches_repository_add():
    record = pool_record("退款几天到账", SOURCE_SELF_CHECK, "证据不足")
    assert record == {
        "raw_question": "退款几天到账", "source": SOURCE_SELF_CHECK, "reason": "证据不足",
    }


def test_refusal_text_is_actionable_and_mentions_human():
    assert "未查到" in REFUSAL_TEXT or "没有查" in REFUSAL_TEXT
    assert "人工" in REFUSAL_TEXT
