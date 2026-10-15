import pytest

from app.db.repository import ChatRepository
from app.tools.business import build_create_ticket, build_query_faq, query_logistics, query_order
from app.tools.registry import build_tools


@pytest.mark.asyncio
async def test_mock_tools_echo_input():
    out = await query_order.ainvoke({"order_id": "1001"})
    assert "1001" in out
    out2 = await query_logistics.ainvoke({"order_id": "1001"})
    assert "1001" in out2


@pytest.mark.asyncio
async def test_query_faq_hit_and_miss(monkeypatch):
    """v2：query_faq 内部改调 rag.pipeline，这里只 pin 工具层的出参契约。"""
    import app.tools.business as business
    from app.rag.pipeline import RagResult

    def fake_pipeline(q, **kw):
        if "退货" in q:
            return RagResult(refused=False, raw_query=q, strategy="hybrid_rerank",
                             evidence_text="[1] 退货退款政策 > 无理由退货\n问：退货\n答：7 天无理由退货")
        return RagResult(refused=True, raw_query=q, strategy="hybrid_rerank",
                         refusal_source="retrieval_low_conf", refusal_reason="召回结果为空")

    monkeypatch.setattr(business, "run_pipeline", fake_pipeline)
    tool = business.build_query_faq(None)
    hit = await tool.ainvoke({"keyword": "退货"})
    miss = await tool.ainvoke({"keyword": "邮费"})
    assert "7 天无理由退货" in hit
    assert "人工" in miss


@pytest.mark.asyncio
async def test_create_ticket_writes_and_escapes(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    tool = build_create_ticket(session_factory, cid)
    out = await tool.ainvoke({"description": "商品坏了", "ticket_type": "售后"})
    assert "工单" in out
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
    assert conv.status.value == "已转人工"


@pytest.mark.asyncio
async def test_build_tools_returns_five_named_tools(session_factory):
    tools = build_tools(session_factory, 1)
    assert [t.name for t in tools] == [
        "query_order", "query_product", "query_logistics", "query_faq", "create_ticket",
    ]


def test_tool_timeout_gives_query_faq_a_longer_budget():
    from app.config import settings
    from app.tools.registry import tool_timeout

    assert tool_timeout("query_faq", 5.0) == settings.rag_tool_timeout
    assert tool_timeout("query_order", 5.0) == 5.0
