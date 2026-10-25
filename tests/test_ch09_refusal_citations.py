from app.rag.pipeline import run_pipeline
from app.rag.strategies import HYBRID_RERANK


class _Embed:
    def embed_documents(self, texts):
        return [[0.0] * 4 for _ in texts]


class _Milvus:
    def hybrid_search(self, *args, **kwargs):
        return [{
            "id": 9, "score": 0.9, "question": "价保", "answer": "下单 7 天内价保",
            "section_path": "商品常见问题 > 价保", "content_type": "policy",
            "row_category": "", "category": "商品常见问题",
        }]


class _Reranker:
    def __init__(self, score=0.2):
        self.score = score

    def rerank(self, query, documents, top_n):
        return [(0, self.score)]


class _Model:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def invoke(self, messages):
        return self.outputs.pop(0)


def test_retrieval_gate_refusal_keeps_candidates_as_citations():
    from app.prompts.rag import RewriteOutput, SelfCheckOutput
    result = run_pipeline(
        "价保规则", strategy=HYBRID_RERANK, embed=_Embed(), milvus=_Milvus(),
        reranker=_Reranker(), model=_Model([
            RewriteOutput(standard_query="价保规则", synonyms=[]),
            SelfCheckOutput(sufficient=True, reason="够"),
        ]),
    )
    assert result.refused is True
    assert result.refusal_source == "retrieval_low_conf"
    assert result.citations
    assert result.citations[0]["chunk_id"] == 9


def test_self_check_refusal_keeps_computed_citations():
    from app.prompts.rag import RewriteOutput, SelfCheckOutput
    result = run_pipeline(
        "价保规则", strategy=HYBRID_RERANK, embed=_Embed(), milvus=_Milvus(),
        reranker=_Reranker(score=0.8), model=_Model([
            RewriteOutput(standard_query="价保规则", synonyms=[]),
            SelfCheckOutput(sufficient=False, reason="不完整"),
        ]),
    )
    assert result.refused is True
    assert result.refusal_source == "self_check"
    assert [c["chunk_id"] for c in result.citations] == [9]
