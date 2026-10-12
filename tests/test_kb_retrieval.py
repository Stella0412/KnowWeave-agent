from app.core.retrieval import RetrievalHit, format_hits, search_knowledge


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]


class FakeMilvus:
    def __init__(self, hits):
        self._hits = hits

    def search(self, vector, top_k):
        return self._hits[:top_k]


def _hit(i, q, a, score):
    return {"id": i, "question": q, "answer": a, "score": score}


def test_search_knowledge_filters_by_threshold_and_keeps_fields():
    hits = [
        _hit(1, "邮费是多少\n运费怎么算", "满 99 包邮", 0.78),
        _hit(2, "退货政策", "7 天无理由", 0.45),
    ]
    rows = search_knowledge(
        "邮费是多少", top_k=5, embeddings=FakeEmbeddings(), milvus=FakeMilvus(hits), min_score=0.55
    )
    assert [r.id for r in rows] == [1]
    assert rows[0].question == "邮费是多少\n运费怎么算"
    assert rows[0].answer == "满 99 包邮"


def test_search_knowledge_returns_empty_below_threshold():
    rows = search_knowledge(
        "今天天气",
        top_k=5,
        embeddings=FakeEmbeddings(),
        milvus=FakeMilvus([_hit(1, "q", "a", 0.1)]),
        min_score=0.55,
    )
    assert rows == []


def test_format_hits_folds_multiline_questions():
    assert format_hits([RetrievalHit(1, "问一\n问二", "答一", 0.9)]) == "问：问一 / 问二\n答：答一"


def test_format_hits_multiple():
    assert (
        format_hits([RetrievalHit(1, "甲", "a", 0.9), RetrievalHit(2, "乙", "b", 0.8)])
        == "问：甲\n答：a\n问：乙\n答：b"
    )
