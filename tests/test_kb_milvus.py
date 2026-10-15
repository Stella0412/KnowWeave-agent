import pytest

from app.kb.milvus_client import MilvusKnowledgeClient

DIM = 16


def unit(axis: int) -> list[float]:
    vec = [0.0] * DIM
    vec[axis] = 1.0
    return vec


def row(i, vector, question="问", answer="答", **meta):
    data = {
        "id": i, "vector": vector, "text": f"{question} {answer}", "question": question,
        "answer": answer, "category": "", "content_type": "md", "section_path": "",
        "row_category": "", "is_key_clause": False,
    }
    data.update(meta)
    return data


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
    store.upsert([row(1, unit(0), "甲", "答甲"), row(2, unit(1), "乙", "答乙")])
    assert store.count() == 2


def test_upsert_same_id_is_idempotent(store):
    store.upsert([row(1, unit(0), "甲", "答甲")])
    store.upsert([row(1, unit(1), "甲", "答甲")])
    assert store.count() == 1


def test_search_returns_id_question_answer_score(store):
    store.upsert([row(10, unit(0), "问甲", "答甲"), row(11, unit(1), "问乙", "答乙")])
    hits = store.search(unit(0), top_k=2)
    assert hits[0]["id"] == 10
    assert hits[0]["question"] == "问甲"
    assert hits[0]["answer"] == "答甲"
    assert hits[0]["score"] > hits[1]["score"]
    assert -1.0001 <= hits[0]["score"] <= 1.0001


def test_delete_ids(store):
    store.upsert([row(1, unit(0), "a", "x"), row(2, unit(1), "b", "y"), row(3, unit(2), "c", "z")])
    store.delete_ids([2])
    assert store.count() == 2


def test_delete_ids_empty_is_noop(store):
    store.upsert([row(1, unit(0), "a", "x")])
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


_CREATE_SCRIPT = """
import sys

from app.kb.milvus_client import MilvusKnowledgeClient

uri, dim = sys.argv[1], int(sys.argv[2])
c = MilvusKnowledgeClient(uri=uri, collection="knowledge", dim=dim)
c.ensure_collection()
vec = [0.0] * dim
vec[0] = 1.0
c.upsert([{"id": 1, "vector": vec, "text": "问甲 答甲", "question": "问甲", "answer": "答甲",
           "category": "", "content_type": "md", "section_path": "", "row_category": "",
           "is_key_clause": False}])
c.close()
"""


def test_search_after_attach_from_another_process(tmp_path):
    """新进程 attach 已存在的 .db 时必须 load，否则 search 报 released。

    真实复现：eval/eval_retrieval.py 首次真跑时 `Collection 'knowledge' is in state
    'released'; call load() before search/get/query`。ensure_collection 在集合已存在时
    直接 return，跳过了 load；而 load 是进程内的（同一进程里再开客户端不会复现，故这里
    用子进程写盘）。
    """
    import subprocess
    import sys
    from pathlib import Path

    uri = str(tmp_path / "kb.db")
    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(
        [sys.executable, "-c", _CREATE_SCRIPT, uri, str(DIM)],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    store = MilvusKnowledgeClient(uri=uri, collection="knowledge", dim=DIM)
    store.ensure_collection()
    try:
        hits = store.search(unit(0), top_k=1)
        assert hits[0]["id"] == 1
    finally:
        store.close()
