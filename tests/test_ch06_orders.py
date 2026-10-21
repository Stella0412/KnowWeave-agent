import pytest
from sqlalchemy import select

from app.db.models import Order, RefundRequest
from app.db.repository import ChatRepository, OrderRepository, RefundRepository
from app.db.seed import seed_orders
from app.tools.business import build_query_order


@pytest.mark.asyncio
async def test_seed_orders_is_idempotent_and_relative(session_factory):
    await seed_orders(session_factory)
    await seed_orders(session_factory)
    async with session_factory() as session:
        rows = list((await session.execute(select(Order).order_by(Order.order_no))).scalars())
    assert [r.order_no for r in rows] == ["1001", "1002", "1003", "1004", "1005"]
    assert rows[0].product_name == "猫爬架"
    assert rows[0].signed_at is not None
    assert rows[2].signed_at is None


@pytest.mark.asyncio
async def test_query_order_reads_table_and_reports_missing(session_factory):
    await seed_orders(session_factory)
    tool = build_query_order(session_factory)
    hit = await tool.ainvoke({"order_id": "1001"})
    assert "猫爬架" in hit
    assert "已签收" in hit
    assert "¥299" in hit
    miss = await tool.ainvoke({"order_id": "9999"})
    assert "未找到订单 9999" in miss


@pytest.mark.asyncio
async def test_refund_repository_creates_and_finds_pending(session_factory):
    await seed_orders(session_factory)
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    async with session_factory() as session:
        row = await RefundRepository(session).create(cid, "1001", "七天无理由")
        await session.commit()
        refund_no = row.refund_no
    assert refund_no.startswith("R")
    async with session_factory() as session:
        found = await RefundRepository(session).get_pending(cid, "1001")
        assert found is not None and found.refund_no == refund_no
