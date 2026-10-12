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
    """v2：query_faq 不再读 faq 表（改走向量检索），故这里只 pin 工具层的出参契约。"""
    import app.tools.business as business
    from app.core.retrieval import RetrievalHit

    monkeypatch.setattr(
        business,
        "search_knowledge",
        lambda q, k: [RetrievalHit(1, "退货政策是什么", "7天无理由退货", 0.9)] if "退货" in q else [],
    )
    tool = build_query_faq(None)  # RAG 路径不用 session，仅验证签名仍收该参数
    hit = await tool.ainvoke({"keyword": "退货"})
    miss = await tool.ainvoke({"keyword": "邮费"})
    assert "7天无理由退货" in hit
    assert "未找到" in miss


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
