import pytest

from app.kb.milvus_client import MilvusKnowledgeClient

DIM = 16


def unit(axis: int, dim: int = DIM) -> list[float]:
    vec = [0.0] * dim
    vec[axis] = 1.0
    return vec


@pytest.fixture
def store(tmp_path):
    s = MilvusKnowledgeClient(uri=str(tmp_path / "kb.db"), collection="knowledge", dim=DIM)
    s.ensure_collection()
    yield s
    s.close()


def _row(i, vector, text, question="问", answer="答", **meta):
    row = {
        "id": i, "vector": vector, "text": text, "question": question, "answer": answer,
        "category": "售后手册", "content_type": "manual",
        "section_path": "售后手册 > 常见问题处理时限", "row_category": "",
        "is_key_clause": False,
    }
    row.update(meta)
    return row


def test_upsert_rows_then_dense_search_carries_metadata(store):
    store.upsert([_row(1, unit(0), "退款 3 个工作日", row_category="退款")])
    hits = store.search(unit(0), top_k=1)
    assert hits[0]["id"] == 1
    assert hits[0]["section_path"] == "售后手册 > 常见问题处理时限"
    assert hits[0]["row_category"] == "退款"
    assert hits[0]["score"] > 0.9


def test_sparse_search_hits_the_keyword_leg(store):
    """验收②的地基：BM25 靠字面词命中，不需要向量像。"""
    store.upsert([
        _row(1, unit(0), "退款 首次响应 2 小时 处理时限 3 个工作日"),
        _row(2, unit(1), "换货 首次响应 2 小时 处理时限 5 个工作日"),
    ])
    hits = store.search_sparse("换货", top_k=2)
    assert hits[0]["id"] == 2
    assert hits[0]["score"] < 0  # BM25 是负数无界分（附录 A.3）


def test_hybrid_search_fuses_and_returns_limit(store):
    store.upsert([
        _row(1, unit(0), "退款 3 个工作日"),
        _row(2, unit(1), "换货 5 个工作日"),
        _row(3, unit(2), "维修 7 个工作日"),
    ])
    hits = store.hybrid_search(unit(0), "换货", top_k=3)
    assert len(hits) == 3
    assert {h["id"] for h in hits} == {1, 2, 3}


def test_hybrid_search_applies_expr_to_both_legs(store):
    store.upsert([
        _row(1, unit(0), "退款 3 个工作日", row_category="退款"),
        _row(2, unit(1), "退款 3 个工作日", row_category="发票"),
    ])
    hits = store.hybrid_search(unit(0), "退款", top_k=5, expr='row_category == "发票"')
    assert [h["id"] for h in hits] == [2]


def test_dense_search_applies_expr(store):
    store.upsert([
        _row(1, unit(0), "甲", content_type="manual"),
        _row(2, unit(0), "乙", content_type="policy"),
    ])
    hits = store.search(unit(0), top_k=5, expr='content_type == "policy"')
    assert [h["id"] for h in hits] == [2]


def test_query_by_expr_returns_raw_rows(store):
    store.upsert([_row(7, unit(0), "退款 3 个工作日", answer="退款 3 个工作日", row_category="退款")])
    rows = store.query_by_expr('row_category == "退款"')
    assert len(rows) == 1
    assert int(rows[0]["id"]) == 7
    assert "3 个工作日" in rows[0]["answer"]


def test_upsert_is_idempotent_by_id(store):
    store.upsert([_row(1, unit(0), "旧")])
    store.upsert([_row(1, unit(1), "新")])
    assert store.count() == 1


def test_drop_collection_then_ensure_recreates_empty(store):
    store.upsert([_row(1, unit(0), "甲")])
    store.drop_collection()
    store.ensure_collection()
    assert store.count() == 0


def test_empty_upsert_is_noop(store):
    store.upsert([])
    assert store.count() == 0
