from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.db.repository import ChatRepository
from app.db.seed import seed_orders
from app.services.chat_service import ChatService, _interrupt_actions, _interrupt_value


def test_interrupt_actions_from_select_order_payload():
    payload = {"kind": "select_order", "orders": [{"order_no": "1001"}]}
    assert _interrupt_actions(payload) == [{
        "type": "select_order",
        "label": "选择订单",
        "prompt": "请选择要退款的订单",
        "orders": [{"order_no": "1001"}],
    }]


class InterruptSnapshot:
    def __init__(self):
        self.values = {"messages": [HumanMessage(content="我要退款")]}
        self.tasks = [SimpleNamespace(interrupts=[SimpleNamespace(
            value={"kind": "select_order", "orders": [{"order_no": "1001"}]}
        )])]


def test_interrupt_value_reads_tasks():
    payload = _interrupt_value(InterruptSnapshot())
    assert payload["kind"] == "select_order"


@pytest.mark.asyncio
async def test_resume_rejects_unknown_order(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    service = ChatService(
        model=object(), intent_model=object(), session_factory=session_factory,
        graph=object(),
    )
    with pytest.raises(ValueError, match="订单不存在"):
        async for _ in service.resume_stream(str(cid), "9999"):
            pass


class NoInterruptGraph:
    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": []}, tasks=[])


@pytest.mark.asyncio
async def test_second_resume_without_pending_interrupt_is_rejected(session_factory):
    await seed_orders(session_factory)
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    service = ChatService(
        model=object(), intent_model=object(), session_factory=session_factory,
        graph=NoInterruptGraph(),
    )
    with pytest.raises(ValueError, match="没有待选择的订单"):
        await service.validate_resume(str(cid), "1001")
