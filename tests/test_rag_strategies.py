import pytest

from app.rag.strategies import (
    BM25,
    HYBRID,
    HYBRID_RERANK,
    STRATEGIES,
    VECTOR,
    build_expr,
    has_rerank,
    is_valid,
    recall,
)


class FakeMilvus:
    def __init__(self):
        self.calls = []

    def search(self, vector, top_k, *, expr=None):
        self.calls.append(("dense", top_k, expr))
        return [{"id": 1, "score": 0.8, "question": "q1", "answer": "a1",
                 "section_path": "文档 > 甲", "content_type": "policy",
                 "row_category": "", "category": "文档"}]

    def search_sparse(self, query_text, top_k, *, expr=None):
        self.calls.append(("sparse", top_k, expr))
        return [{"id": 2, "score": -3.0, "question": "q2", "answer": "a2",
                 "section_path": "文档 > 乙", "content_type": "policy",
                 "row_category": "", "category": "文档"}]

    def hybrid_search(self, dense_vector, sparse_query_text, top_k, *, rrf_k=60, expr=None):
        self.calls.append(("hybrid", top_k, expr, rrf_k))
        return [
            {"id": 2, "score": 0.033, "question": "q2", "answer": "a2",
             "section_path": "文档 > 乙", "content_type": "policy",
             "row_category": "", "category": "文档"},
            {"id": 1, "score": 0.032, "question": "q1", "answer": "a1",
             "section_path": "文档 > 甲", "content_type": "policy",
             "row_category": "", "category": "文档"},
        ]


def test_strategy_catalog_and_rerank_flag():
    assert STRATEGIES == (VECTOR, BM25, HYBRID, HYBRID_RERANK)
    assert [has_rerank(s) for s in STRATEGIES] == [False, False, False, True]
    assert is_valid("没有这个策略") is False


def test_build_expr_none_and_empty_give_no_filter():
    assert build_expr(None) is None
    assert build_expr({}) is None


def test_build_expr_equality_and_in_list():
    assert build_expr({"row_category": "退款"}) == 'row_category == "退款"'
    assert build_expr({"content_type": ["policy", "manual"]}) == 'content_type in ["policy", "manual"]'


def test_build_expr_combines_fields_with_and():
    expr = build_expr({"row_category": "退款", "content_type": "manual"})
    assert expr == 'row_category == "退款" and content_type == "manual"'


def test_build_expr_rejects_unknown_field():
    """过滤字段是白名单：不把调用方给的名字直接拼进表达式（spec §8）。"""
    with pytest.raises(ValueError):
        build_expr({"用户随便写的字段": "x"})


def test_build_expr_escapes_quotes_in_values():
    assert build_expr({"row_category": '退款" or 1==1'}) == 'row_category == "退款\\" or 1==1"'


def test_build_expr_escapes_backslash():
    assert build_expr({"row_category": "a\\b"}) == 'row_category == "a\\\\b"'


def test_recall_vector_uses_dense_leg_only():
    milvus = FakeMilvus()
    items = recall(VECTOR, dense_vector=[0.1], bm25_query="运费", top_k=50, rrf_k=60, milvus=milvus)
    assert [i.id for i in items] == [1]
    assert milvus.calls == [("dense", 50, None)]


def test_recall_bm25_uses_sparse_leg_only():
    milvus = FakeMilvus()
    items = recall(BM25, dense_vector=[0.1], bm25_query="运费", top_k=50, rrf_k=60, milvus=milvus)
    assert [i.id for i in items] == [2]
    assert items[0].score == -3.0
    assert milvus.calls == [("sparse", 50, None)]


def test_recall_hybrid_uses_hybrid_search_with_rrf_k():
    milvus = FakeMilvus()
    items = recall(HYBRID, dense_vector=[0.1], bm25_query="运费", top_k=50, rrf_k=77, milvus=milvus)
    assert [i.id for i in items] == [2, 1]
    assert milvus.calls == [("hybrid", 50, None, 77)]


def test_recall_hybrid_rerank_uses_same_recall_as_hybrid():
    """排序是后面精排的事，召回段两条策略必须完全一样（否则对比报告不可信）。"""
    milvus = FakeMilvus()
    recall(HYBRID, dense_vector=[0.1], bm25_query="运费", top_k=50, rrf_k=60, milvus=milvus)
    first = milvus.calls[:]
    milvus.calls.clear()
    recall(HYBRID_RERANK, dense_vector=[0.1], bm25_query="运费", top_k=50, rrf_k=60, milvus=milvus)
    assert milvus.calls == first


def test_recall_passes_filter_expr_to_every_leg():
    milvus = FakeMilvus()
    recall(HYBRID, dense_vector=[0.1], bm25_query="运费", top_k=50, rrf_k=60,
           filters={"row_category": "退款"}, milvus=milvus)
    assert milvus.calls[0][2] == 'row_category == "退款"'


def test_recall_unknown_strategy_raises():
    with pytest.raises(ValueError):
        recall("没有这个策略", dense_vector=[0.1], bm25_query="q", top_k=5, rrf_k=60, milvus=FakeMilvus())


def test_recall_propagates_milvus_failure():
    """召回抛错要让上层看见（上层统一包成 UNAVAILABLE），不在这一层吞。"""

    class Boom:
        def search(self, *a, **kw):
            raise RuntimeError("milvus 连不上")

    with pytest.raises(RuntimeError):
        recall(VECTOR, dense_vector=[0.1], bm25_query="q", top_k=5, rrf_k=60, milvus=Boom())
