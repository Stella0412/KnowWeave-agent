import asyncio

import pytest
from sqlalchemy import func, select

from app.db.models import Conversation, Ticket
from app.tools.ticket_confirmation import TicketDraft, TicketDraftConflict, create_ticket_once


@pytest.fixture
async def conversation_id(session_factory):
    async with session_factory() as session:
        row = Conversation(user_id="ticket-test")
        session.add(row)
        await session.commit()
        return row.id


def _draft(conversation_id: int, description: str = "猫粮包装破损") -> TicketDraft:
    return TicketDraft.create(
        conversation_id=conversation_id,
        tool_call_id="call-1",
        ticket_type="售后",
        description=description,
        request_message_id=10,
    )


@pytest.mark.asyncio
async def test_same_call_creates_one_ticket(session_factory, conversation_id):
    draft = _draft(conversation_id)
    first = await create_ticket_once(session_factory, draft)
    second = await create_ticket_once(session_factory, draft)

    assert first == second
    assert first.startswith("T8") and len(first) == 32
    async with session_factory() as session:
        count = await session.scalar(
            select(func.count()).select_from(Ticket).where(Ticket.ticket_no == first)
        )
    assert count == 1


@pytest.mark.asyncio
async def test_same_identity_with_changed_content_is_conflict(session_factory, conversation_id):
    await create_ticket_once(session_factory, _draft(conversation_id))

    with pytest.raises(TicketDraftConflict):
        await create_ticket_once(session_factory, _draft(conversation_id, "篡改后的内容"))


@pytest.mark.asyncio
async def test_concurrent_submit_creates_one_ticket(session_factory, conversation_id):
    draft = _draft(conversation_id)
    first, second = await asyncio.gather(
        create_ticket_once(session_factory, draft),
        create_ticket_once(session_factory, draft),
    )

    assert first == second
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Ticket)) == 1
