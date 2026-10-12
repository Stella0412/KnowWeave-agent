import pytest

from app.db.repository import ChatRepository
from app.db.seed import seed_faq


@pytest.mark.asyncio
async def test_seed_faq_idempotent(session_factory):
    await seed_faq(session_factory)
    await seed_faq(session_factory)  # 重复执行不重复插入
    async with session_factory() as session:
        rows = await ChatRepository(session).search_faq("退货")
        postage = await ChatRepository(session).search_faq("邮费")
    assert len(rows) >= 1
    assert postage == []  # 刻意不含邮费/运费，为验收③保留漏召回


@pytest.mark.asyncio
async def test_seed_conversations_is_idempotent(session_factory):
    from sqlalchemy import select

    from app.db.models import Conversation, Message
    from app.db.seed import seed_conversations

    await seed_conversations(session_factory)
    await seed_conversations(session_factory)
    async with session_factory() as s:
        convs = (await s.execute(select(Conversation))).scalars().all()
        msgs = (await s.execute(select(Message))).scalars().all()
    assert len(convs) == 2
    assert len(msgs) == 8
    assert {m.role.value for m in msgs} == {"user", "assistant"}
