import pytest

from app.db.models import ConversationStatus
from app.db.repository import ChatRepository
from app.db.seed import seed_orders
from app.tools.business import (
    build_create_ticket,
    build_query_faq,
    build_query_order,
    create_ticket_record,
    query_logistics,
)
from app.tools.registry import build_tools


@pytest.mark.asyncio
async def test_query_order_reads_seeded_table(session_factory):
    await seed_orders(session_factory)
    out = await build_query_order(session_factory).ainvoke({"order_id": "1001"})
    assert "1001" in out
    assert "物流单号=SF1001CN" in out


def test_query_logistics_schema_requires_tracking_no():
    with pytest.raises(Exception):
        query_logistics.args_schema.model_validate({"order_id": "1001"})


@pytest.mark.asyncio
async def test_query_faq_hit_and_miss(monkeypatch):
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
async def test_create_ticket_writes_and_escalates(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    tool = build_create_ticket(session_factory, cid)
    out = await tool.ainvoke({"description": "商品坏了", "ticket_type": "售后"})
    assert "工单" in out
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
    assert conv.status is ConversationStatus.ESCALATED


@pytest.mark.asyncio
async def test_create_ticket_record_can_skip_escalation(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    ticket_no = await create_ticket_record(
        session_factory, cid, "商品破损", "售后", escalate=False
    )
    assert ticket_no.startswith("T")
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
    assert conv.status is ConversationStatus.ACTIVE


@pytest.mark.asyncio
async def test_build_tools_has_six_tools_in_static_order(session_factory):
    tools = build_tools(session_factory, 1)
    assert [t.name for t in tools] == [
        "query_order", "query_product", "query_logistics",
        "query_faq", "submit_refund", "create_ticket",
    ]


def test_tool_timeout_gives_query_faq_a_longer_budget(monkeypatch):
    from app.config import settings
    from app.tools.registry import tool_timeout

    monkeypatch.setattr(settings, "rag_tool_timeout", 7.5)
    assert tool_timeout("query_faq", 5.0) == 7.5
    assert tool_timeout("query_order", 5.0) == 5.0
