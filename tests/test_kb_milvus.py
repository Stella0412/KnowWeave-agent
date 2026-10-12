import pytest

from app.kb.milvus_client import MilvusKnowledgeClient

DIM = 16


def unit(axis: int) -> list[float]:
    vec = [0.0] * DIM
    vec[axis] = 1.0
    return vec


@pytest.fixture
def store(tmp_path):
    s = MilvusKnowledgeClient(uri=str(tmp_path / "kb.db"), collection="knowledge", dim=DIM)
    s.ensure_collection()
    yield s
    s.close()


def test_ensure_collection_is_idempotent(store):
    store.ensure_collection()
    assert store.count() == 0


def test_upsert_then_count(store):
    store.upsert([1, 2], [unit(0), unit(1)], ["甲", "乙"], ["答甲", "答乙"])
    assert store.count() == 2


def test_upsert_same_id_is_idempotent(store):
    store.upsert([1], [unit(0)], ["甲"], ["答甲"])
    store.upsert([1], [unit(1)], ["甲"], ["答甲"])
    assert store.count() == 1


def test_search_returns_id_question_answer_score(store):
    store.upsert([10, 11], [unit(0), unit(1)], ["问甲", "问乙"], ["答甲", "答乙"])
    hits = store.search(unit(0), top_k=2)
    assert hits[0]["id"] == 10
    assert hits[0]["question"] == "问甲"
    assert hits[0]["answer"] == "答甲"
    assert hits[0]["score"] > hits[1]["score"]
    assert -1.0001 <= hits[0]["score"] <= 1.0001


def test_delete_ids(store):
    store.upsert([1, 2, 3], [unit(0), unit(1), unit(2)], ["a", "b", "c"], ["x", "y", "z"])
    store.delete_ids([2])
    assert store.count() == 2


def test_delete_ids_empty_is_noop(store):
    store.upsert([1], [unit(0)], ["a"], ["x"])
    store.delete_ids([])
    assert store.count() == 1


def test_build_milvus_client_creates_parent_dir(tmp_path):
    from app.config import Settings

    from app.kb.milvus_client import build_milvus_client

    target = tmp_path / "nested" / "data" / "milvus.db"
    s = Settings(_env_file=None, milvus_uri=str(target))
    store = build_milvus_client(s)
    try:
        store.ensure_collection()
        assert target.parent.is_dir()
    finally:
        store.close()
