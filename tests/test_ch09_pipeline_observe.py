import asyncio

from app.observability.context import TurnObserver, observer_scope
from app.rag.pipeline import run_pipeline
from app.rag.strategies import HYBRID_RERANK


class _Embed:
    def embed_documents(self, texts):
        return [[0.0] * 4 for _ in texts]


class _Milvus:
    def hybrid_search(self, *args, **kwargs):
        return [{"id": 1, "score": 0.9, "question": "运费", "answer": "满 99 包邮",
                 "section_path": "配送 > 运费", "content_type": "policy",
                 "row_category": "", "category": "配送"}]


class _Reranker:
    def rerank(self, query, documents, top_n):
        return [(0, 0.8)]


class _Model:
    def invoke(self, messages):
        from app.prompts.rag import RewriteOutput, SelfCheckOutput
        if not getattr(self, "n", 0):
            self.n = 1
            return RewriteOutput(standard_query="运费", synonyms=[])
        return SelfCheckOutput(sufficient=True, reason="够")


class FakeHandler:
    def __init__(self):
        self.stages = []

    def record_stage(self, token, kind, *, model, duration_ms, texts=None, items=0):
        self.stages.append((token, kind, model, items))


def test_embed_and_rerank_are_recorded():
    handler = FakeHandler()
    observer = TurnObserver(handler=handler, turn_token="t")
    with observer_scope(observer):
        result = run_pipeline(
            "运费怎么算", strategy=HYBRID_RERANK, embed=_Embed(),
            milvus=_Milvus(), reranker=_Reranker(), model=_Model(),
        )
    assert result.refused is False
    assert [s[1] for s in handler.stages] == ["embedding", "rerank"]
    assert handler.stages[1][3] == 1


def test_to_thread_copies_observer_context():
    handler = FakeHandler()
    observer = TurnObserver(handler=handler, turn_token="t")

    def work():
        from app.observability.context import get_observer
        assert get_observer() is observer
        return 1

    with observer_scope(observer):
        assert asyncio.run(asyncio.to_thread(work)) == 1
