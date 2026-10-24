import asyncio

import pytest
from sqlalchemy import func, select

from app.db.models import Conversation, Ticket
from app.tools.ticket_confirmation import TicketDraft, TicketDraftConflict, create_ticket_once
from app.tools.ticket_confirmation import ticket_no_for
from app.db.base import Base
from app.db.models import TicketType, TicketStatus
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.exc import IntegrityError


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


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
async def test_database_collision_after_absent_lookup(tmp_path, changed):
    """A second independent session wins between lookup and autoflush."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'collision.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with factory() as session:
            conversation = Conversation(user_id="collision")
            session.add(conversation)
            await session.commit()
            draft = _draft(conversation.id)
        collisions = []

        class RacingSession(AsyncSession):
            async def get(self, entity, ident, **kwargs):
                result = await super().get(entity, ident, **kwargs)
                if entity is Ticket and result is None and not collisions:
                    collisions.append(ident)
                    async with factory() as competitor:
                        competitor.add(Ticket(
                            ticket_no=ident, conversation_id=draft.conversation_id,
                            description="different" if changed else draft.description,
                            ticket_type=TicketType(draft.ticket_type), status=TicketStatus.PENDING,
                        ))
                        await competitor.commit()
                return result

        racing_factory = async_sessionmaker(engine, class_=RacingSession, expire_on_commit=False)
        if changed:
            with pytest.raises(TicketDraftConflict):
                await create_ticket_once(racing_factory, draft)
        else:
            assert await create_ticket_once(racing_factory, draft) == ticket_no_for(draft)
        assert collisions == [ticket_no_for(draft)]
        async with factory() as session:
            assert await session.scalar(select(func.count()).select_from(Ticket)) == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_unrelated_integrity_failure_preserves_original(session_factory, conversation_id):
    class InvalidInsertSession(AsyncSession):
        def add(self, instance, **kwargs):
            if isinstance(instance, Ticket):
                instance.description = None
            return super().add(instance, **kwargs)

    factory = async_sessionmaker(session_factory.kw["bind"], class_=InvalidInsertSession)
    with pytest.raises(IntegrityError, match="NOT NULL constraint failed") as caught:
        await create_ticket_once(factory, _draft(conversation_id))
    assert caught.value.orig is not None
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Ticket)) == 0
        assert (await session.get(Conversation, conversation_id)).status.value != "已转人工"


@pytest.mark.asyncio
async def test_finished_ticket_lock_is_released(session_factory, conversation_id):
    import gc
    from app.tools.ticket_confirmation import _ticket_locks
    draft = _draft(conversation_id)
    await create_ticket_once(session_factory, draft)
    gc.collect()  # Exception tracebacks from preceding cases may retain frames until GC.
    assert ticket_no_for(draft) not in _ticket_locks
