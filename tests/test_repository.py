import pytest

from app.db.models import ConversationStatus, TicketType
from app.db.repository import ChatRepository


@pytest.mark.asyncio
async def test_create_conversation_defaults(session_factory):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        await session.commit()
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
    assert conv.user_id == "anonymous"
    assert conv.status == ConversationStatus.ACTIVE


@pytest.mark.asyncio
async def test_add_and_list_messages_ordered(session_factory):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        await repo.add_message(cid, role="user", content="你好")
        await repo.add_message(cid, role="assistant", content="您好")
        await session.commit()
        rows = await repo.list_messages(cid)
    assert [r.role.value for r in rows] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_search_faq_like(session_factory):
    from app.db.models import FAQ
    async with session_factory() as session:
        session.add(FAQ(question="退货政策是什么", answer="7天无理由退货", category="售后"))
        session.add(FAQ(question="发货时间多久", answer="48小时内发货", category="物流"))
        await session.commit()
        hits = await ChatRepository(session).search_faq("退货")
        miss = await ChatRepository(session).search_faq("邮费")
    assert len(hits) == 1
    assert hits[0].question == "退货政策是什么"
    assert miss == []


@pytest.mark.asyncio
async def test_create_ticket_generates_no_and_escapes(session_factory):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        no = await repo.create_ticket(cid, "商品坏了", TicketType.AFTER_SALE)
        await repo.set_conversation_status(cid, ConversationStatus.ESCALATED)
        await session.commit()
    assert no.startswith("T") and len(no) == len("T") + 8 + 3
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
    assert conv.status == ConversationStatus.ESCALATED
