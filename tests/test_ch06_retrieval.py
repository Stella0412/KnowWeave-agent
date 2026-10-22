import app.rag.pipeline as pipeline
from app.rag.evidence import Evidence


def _ev(eid, score, question="q"):
    return Evidence(
        id=eid, question=question, answer=f"a{eid}", section_path="政策 > 退货",
        content_type="text", row_category="", category="售后", score=score,
    )


def test_expanded_pipeline_dedupes_by_id_keeps_max_and_reranks_once(monkeypatch):
    calls = {"recall": [], "rerank": [], "self_check": 0}

    def fake_recall(strategy, *, dense_vector, bm25_query, top_k, rrf_k, filters=None, milvus=None):
        calls["recall"].append(bm25_query)
        if bm25_query == "q1":
            return [_ev(1, 0.2), _ev(2, 0.4)]
        return [_ev(1, 0.9), _ev(3, 0.3)]

    def fake_rerank(candidates, query, *, reranker=None, top_n=None):
        calls["rerank"].append((query, [c.id for c in candidates]))
        return candidates, 0.9, False

    def fake_self_check(query, evidence, *, model=None):
        calls["self_check"] += 1
        return True, ""

    monkeypatch.setattr(pipeline, "embed_query", lambda q, embeddings=None: [float(len(q))])
    monkeypatch.setattr(pipeline, "recall", fake_recall)
    monkeypatch.setattr(pipeline, "rerank_stage", fake_rerank)
    monkeypatch.setattr(pipeline, "_self_check", fake_self_check)

    result = pipeline.run_expanded_pipeline(
        ["q1", "q2"], "订单1001能退吗", strategy="hybrid_rerank"
    )
    assert result.refused is False
    # assemble() 按首尾序重排：3 条 → 原下标 [0, 2, 1]
    assert [item.id for item in result.items] == [1, 3, 2]
    assert {item.id: item.score for item in result.items} == {1: 0.9, 2: 0.4, 3: 0.3}
    assert calls["recall"] == ["q1", "q2"]
    # 精排用扩写主查询（第 1 条），不用口语原句：原句含商品名/订单号，政策条款分会被压到闸门下。
    assert calls["rerank"][0][0] == "q1"
    assert calls["rerank"][0][1] == [1, 2, 3]
    assert calls["self_check"] == 1


def test_expanded_pipeline_uses_primary_expansion_query_for_rerank(monkeypatch):
    """口语原句精排会低于拒答阈值；用扩写主查询才应该通过闸门。"""

    def fake_recall(strategy, *, dense_vector, bm25_query, top_k, rrf_k, filters=None, milvus=None):
        return [_ev(1, 0.2)]

    def fake_rerank(candidates, query, *, reranker=None, top_n=None):
        # 模拟真实行为：政策化查询分数高，含商品名/订单号的口语原句分数低。
        return candidates, (0.8 if query == "核心政策查询" else 0.1), False

    def fake_self_check(query, evidence, *, model=None):
        assert query == "订单1001的猫爬架能退吗"
        return True, ""

    monkeypatch.setattr(pipeline, "embed_query", lambda q, embeddings=None: [float(len(q))])
    monkeypatch.setattr(pipeline, "recall", fake_recall)
    monkeypatch.setattr(pipeline, "rerank_stage", fake_rerank)
    monkeypatch.setattr(pipeline, "_self_check", fake_self_check)

    result = pipeline.run_expanded_pipeline(
        ["核心政策查询", "时限查询", "例外查询"],
        "订单1001的猫爬架能退吗",
        strategy="hybrid_rerank",
    )

    assert result.refused is False
    assert result.rerank_score == 0.8
