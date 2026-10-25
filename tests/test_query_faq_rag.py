import app.tools.business as business
from app.core.retrieval import UNAVAILABLE
from app.rag.evidence import Evidence
from app.rag.pipeline import RagResult, TurnContext


def _result(**over):
    payload = {
        "refused": False, "raw_query": "邮费是多少", "strategy": "hybrid_rerank",
        "evidence_text": "[1] 开票与配送 > 运费与包邮\n问：运费怎么算\n答：满 99 元包邮",
        "citations": [{"n": 1, "chunk_id": 7, "answer": "满 99 元包邮"}],
        "items": [Evidence(id=7, question="运费怎么算", answer="满 99 元包邮",
                           section_path="开票与配送 > 运费与包邮", content_type="policy",
                           row_category="", category="开票与配送", score=0.8, n=1)],
    }
    payload.update(over)
    return RagResult(**payload)


async def test_query_faq_returns_evidence_text_and_collects_citations(monkeypatch):
    monkeypatch.setattr(business, "run_pipeline", lambda q, **kw: _result())
    turn = TurnContext(raw_question="邮费是多少", conversation_id=3)
    tool = business.build_query_faq(session_factory=None, turn=turn)
    out = await tool.ainvoke({"keyword": "邮费是多少"})
    assert "满 99 元包邮" in out
    assert turn.citations[0]["chunk_id"] == 7
    # 入参契约不变
    assert list(tool.args_schema.model_fields) == ["keyword"]


async def test_query_faq_refusal_returns_refusal_text_and_writes_pool(monkeypatch):
    calls = []

    class FakeFlywheel:
        def __init__(self, session_factory):
            pass

        async def record_event_isolated(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(business, "run_pipeline", lambda q, **kw: _result(
        refused=True, refusal_source="retrieval_low_conf",
        refusal_reason="召回结果为空", evidence_text="", citations=[], items=[],
    ))
    monkeypatch.setattr(business, "FlywheelService", FakeFlywheel)

    turn = TurnContext(raw_question="今天天气怎么样", conversation_id=9)
    tool = business.build_query_faq(session_factory=object, turn=turn)
    out = await tool.ainvoke({"keyword": "今天天气怎么样"})
    assert "「今天天气怎么样」" in out
    assert "暂时" in out
    assert "知识库" not in out
    assert calls == [{
        "conversation_id": 9, "raw_question": "今天天气怎么样",
        "source": "retrieval_low_conf", "reason": "召回结果为空", "citations": [],
    }]
    assert turn.citations == []


async def test_query_faq_pool_write_failure_does_not_break_the_answer(monkeypatch):
    """池子是后台资产，不该拖垮会话（Review Focus 6）。"""

    class BoomFactory:
        def __call__(self):
            raise RuntimeError("MySQL 挂了")

    monkeypatch.setattr(business, "run_pipeline", lambda q, **kw: _result(
        refused=True, refusal_source="self_check", refusal_reason="证据不足",
        evidence_text="", citations=[], items=[],
    ))

    tool = business.build_query_faq(
        session_factory=BoomFactory(), turn=TurnContext(raw_question="x")
    )
    out = await tool.ainvoke({"keyword": "x"})
    assert "「x」" in out


async def test_query_faq_does_not_raise_on_pipeline_failure(monkeypatch):
    def boom(q, **kw):
        raise RuntimeError("milvus 连不上")

    monkeypatch.setattr(business, "run_pipeline", boom)
    tool = business.build_query_faq(session_factory=None, turn=TurnContext(raw_question="x"))
    assert await tool.ainvoke({"keyword": "邮费是多少"}) == UNAVAILABLE


async def test_query_faq_without_turn_still_works(monkeypatch):
    """没有 TurnContext（脚本/单测直调工具）时不能崩，只是收不到引用。"""
    monkeypatch.setattr(business, "run_pipeline", lambda q, **kw: _result())
    tool = business.build_query_faq(session_factory=None)
    assert "满 99 元包邮" in await tool.ainvoke({"keyword": "邮费是多少"})
