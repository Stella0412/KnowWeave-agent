import pytest

from app.prompts.rag import RewriteOutput, SelfCheckOutput
from app.rag.evidence import Evidence
from app.rag.pipeline import TurnContext, run_pipeline
from app.rag.rerank import RerankError
from app.rag.strategies import BM25, HYBRID, HYBRID_RERANK, VECTOR

_QUERY_MAX = 2000


class FakeModel:
    """按调用序返回预置输出；boom 用来制造上游失败。"""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        item = self._outputs.pop(0) if self._outputs else None
        if isinstance(item, Exception):
            raise item
        if item is None:
            raise RuntimeError("FakeModel 没有更多输出")
        return item


class FakeMilvus:
    def __init__(self, hits=None, boom=None):
        self.hits = hits if hits is not None else [
            {"id": 1, "score": 0.9, "question": "运费", "answer": "满 99 包邮",
             "section_path": "开票与配送 > 运费与包邮", "content_type": "policy",
             "row_category": "", "category": "开票与配送"},
            {"id": 2, "score": 0.7, "question": "退款", "answer": "以渠道为准",
             "section_path": "退货退款政策 > 无理由退货 > 退款时效", "content_type": "policy",
             "row_category": "", "category": "退货退款政策"},
        ]
        self.boom = boom
        self.calls = []

    def _leg(self, kind, *args, **kwargs):
        self.calls.append((kind, kwargs.get("expr")))
        if self.boom:
            raise self.boom
        return self.hits

    def search(self, vector, top_k, *, expr=None):
        return self._leg("dense", expr=expr)

    def search_sparse(self, query_text, top_k, *, expr=None):
        return self._leg("sparse", expr=expr)

    def hybrid_search(self, dense_vector, sparse_query_text, top_k, *, rrf_k=60, expr=None):
        return self._leg("hybrid", expr=expr)


class FakeReranker:
    def __init__(self, scores=None, boom=None):
        self.scores = scores if scores is not None else [0.8, 0.2]
        self.boom = boom
        self.calls = []

    def rerank(self, query, documents, top_n):
        self.calls.append((query, documents, top_n))
        if self.boom:
            raise self.boom
        ranked = sorted(enumerate(self.scores), key=lambda p: p[1], reverse=True)
        return ranked[:top_n]


class _Embed:
    """替身必须长得像真实嵌入对象（Ruling R15）：真实链路 `app/core/embeddings.py:36-37`
    是 `emb.embed_documents(texts)`，不是把 embeddings 当函数调用。原稿写成普通函数
    `def _embed(texts)`，run_pipeline 走到 `embed_texts` 时会去调 `.embed_documents` 而炸
    （非空 query 的用例全挂）。调用点 `embed=_embed` 六处一字不改。"""

    def embed_documents(self, texts):
        return [[0.0] * 4 for _ in texts]


_embed = _Embed()


def _run(**over):
    payload = {
        "strategy": HYBRID_RERANK,
        "embed": _embed,
        "milvus": FakeMilvus(),
        "reranker": FakeReranker(),
        "model": FakeModel([
            RewriteOutput(standard_query="运费怎么算", synonyms=["邮费"]),
            SelfCheckOutput(sufficient=True, reason="资料里有运费规则"),
        ]),
        "source_files": {"开票与配送": "billing-shipping.md"},
    }
    payload.update(over)
    return run_pipeline("邮费怎么算", **payload)


def test_happy_path_returns_numbered_evidence_and_citations():
    result = _run()
    assert result.refused is False
    assert result.evidence_text.startswith("[1]")
    assert [c["n"] for c in result.citations] == [1, 2]
    assert result.citations[0]["source_file"] == "billing-shipping.md"
    assert result.standard_query == "运费怎么算"
    assert result.strategy == HYBRID_RERANK


def test_head_tail_numbering_matches_assembly_order():
    result = _run()
    # 精排把 id=1(0.8) 排第一、id=2(0.2) 排第二；编号按装配序 = 原精排序
    assert [c["chunk_id"] for c in result.citations] == [1, 2]


def test_rerank_failure_degrades_to_rrf_order():
    """Ruling R1 / R4：精排挂掉要退化成 hybrid 顺序、**不崩**，而且**不许拿 RRF 分去比精排阈值**。

    所以这里刻意把召回分设成 RRF 名次倒数的量级（≈0.033 / 0.016）：若降级时还把
    `max(候选分)` 当精排分去比 `rerank_min_score=0.10`，这条用例会因为「整轮被拒答」而失败
    —— 这正是它存在的意义（用默认的 0.9/0.7 假分则无论修没修都会通过，钉不住任何东西）。
    """
    rrf_shaped = [
        {"id": 1, "score": 0.0328, "question": "运费", "answer": "满 99 包邮",
         "section_path": "开票与配送 > 运费与包邮", "content_type": "policy",
         "row_category": "", "category": "开票与配送"},
        {"id": 2, "score": 0.0164, "question": "退款", "answer": "以渠道为准",
         "section_path": "退货退款政策 > 无理由退货 > 退款时效", "content_type": "policy",
         "row_category": "", "category": "退货退款政策"},
    ]
    result = _run(
        reranker=FakeReranker(boom=RerankError("上游 503")),
        milvus=FakeMilvus(hits=rrf_shaped),
    )
    assert result.refused is False
    assert result.rerank_degraded is True
    assert [c["chunk_id"] for c in result.citations] == [1, 2]   # 用召回顺序，不崩


def test_rewrite_failure_degrades_to_raw_query():
    result = _run(model=FakeModel([
        RuntimeError("改写上游 500"),
        SelfCheckOutput(sufficient=True, reason="够"),
    ]))
    assert result.rewrite_degraded is True
    assert result.standard_query == "邮费怎么算"
    assert result.refused is False


def test_self_check_failure_does_not_refuse():
    """自评失败当成够答：一次网络抖动不该把能答的问题拒掉（Review Focus 3）。"""
    result = _run(model=FakeModel([
        RewriteOutput(standard_query="运费怎么算", synonyms=[]),
        RuntimeError("自评上游 500"),
    ]))
    assert result.refused is False


def test_self_check_insufficient_refuses_and_blocks_evidence():
    result = _run(model=FakeModel([
        RewriteOutput(standard_query="运费怎么算", synonyms=[]),
        SelfCheckOutput(sufficient=False, reason="资料里没有到账天数的依据"),
    ]))
    assert result.refused is True
    assert result.refusal_source == "self_check"
    assert result.evidence_text == ""      # 拒答时不把证据塞进 prompt
    assert result.citations == []
    assert "到账天数" in result.refusal_reason


def test_self_check_receives_user_raw_words_not_the_rewritten_query():
    model = FakeModel([
        RewriteOutput(standard_query="运费怎么算", synonyms=[]),
        SelfCheckOutput(sufficient=True, reason="够"),
    ])
    _run(model=model)
    self_check_prompt = model.calls[1][1].content
    assert "邮费怎么算" in self_check_prompt


def test_empty_recall_refuses_via_retrieval_gate():
    result = _run(milvus=FakeMilvus(hits=[]))
    assert result.refused is True
    assert result.refusal_source == "retrieval_low_conf"
    assert result.citations == []


def test_low_rerank_score_refuses():
    result = _run(reranker=FakeReranker(scores=[0.01, 0.005]))
    assert result.refused is True
    assert result.refusal_source == "retrieval_low_conf"


def test_milvus_failure_propagates_for_caller_to_map_to_unavailable():
    """Milvus 挂了不是「拒答」，是「知识库不可用」：区别必须传出去（Review Focus 1）。"""
    with pytest.raises(RuntimeError):
        _run(milvus=FakeMilvus(boom=RuntimeError("milvus 连不上")))


def test_empty_query_refuses_without_calling_upstream():
    model = FakeModel([])
    result = run_pipeline(
        "   ", strategy=HYBRID_RERANK, embed=_embed, milvus=FakeMilvus(),
        reranker=FakeReranker(), model=model,
    )
    assert result.refused is True
    assert result.citations == []
    assert model.calls == []           # 空 query 不该打上游（Review Focus 4）


def test_overlong_query_is_truncated_before_retrieval():
    milvus = FakeMilvus()
    model = FakeModel([
        RewriteOutput(standard_query="长问题", synonyms=[]),
        SelfCheckOutput(sufficient=True, reason="够"),
    ])
    long_query = "长" * 5000
    result = run_pipeline(
        long_query, strategy=VECTOR, embed=_embed, milvus=milvus,
        reranker=FakeReranker(), model=model,
    )
    assert result.refused is False
    assert len(milvus.calls) == 1


def test_bm25_strategy_never_touches_dense_leg():
    milvus = FakeMilvus()
    model = FakeModel([
        RewriteOutput(standard_query="运费", synonyms=[]),
        SelfCheckOutput(sufficient=True, reason="够"),
    ])
    result = run_pipeline(
        "邮费怎么算", strategy=BM25, embed=_embed, milvus=milvus,
        reranker=FakeReranker(), model=model,
    )
    assert [c["chunk_id"] for c in result.citations] == [1, 2]
    assert [c[0] for c in milvus.calls] == ["sparse"]


def test_vector_strategy_applies_dense_threshold():
    milvus = FakeMilvus(hits=[{"id": 1, "score": 0.2, "question": "q", "answer": "a",
                               "section_path": "文档 > 节", "content_type": "md",
                               "row_category": "", "category": "文档"}])
    model = FakeModel([
        RewriteOutput(standard_query="q", synonyms=[]),
        SelfCheckOutput(sufficient=True, reason="够"),
    ])
    result = run_pipeline(
        "q", strategy=VECTOR, embed=_embed, milvus=milvus, reranker=FakeReranker(), model=model,
    )
    assert result.refused is True
    assert result.refusal_source == "retrieval_low_conf"


def test_filters_reach_the_retrieval_leg():
    milvus = FakeMilvus()
    model = FakeModel([
        RewriteOutput(standard_query="q", synonyms=[]),
        SelfCheckOutput(sufficient=True, reason="够"),
    ])
    run_pipeline(
        "q", strategy=HYBRID, filters={"row_category": "退款"}, embed=_embed,
        milvus=milvus, reranker=FakeReranker(), model=model,
    )
    assert milvus.calls[0][1] == 'row_category == "退款"'


def test_turn_context_collects_citations():
    ctx = TurnContext(raw_question="邮费怎么算", conversation_id=7)
    result = _run()
    ctx.citations.extend(result.citations)
    assert ctx.citations[0]["n"] == 1
    assert ctx.conversation_id == 7
