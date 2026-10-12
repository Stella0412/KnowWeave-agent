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


@pytest.mark.asyncio
async def test_seed_conversations_ignores_unrelated_existing_conversations(session_factory):
    """本机已有 ch02 真实会话时，演示会话仍要灌进去。

    否则（旧判据「有任何会话就直接返回」）在用过 ch02 的库上演示会话静默缺失，
    mine_qa 挖不到「发票/快递」这类只存在于演示会话里的问答。
    """
    from sqlalchemy import select

    from app.db.models import Conversation, Message, MessageRole
    from app.db.seed import DEMO_CONVERSATIONS, seed_conversations

    async with session_factory() as s:
        conv = Conversation(user_id="someone")
        s.add(conv)
        await s.flush()
        s.add(Message(conversation_id=conv.id, role=MessageRole.USER, content="订单1001到哪里"))
        await s.commit()

    await seed_conversations(session_factory)
    await seed_conversations(session_factory)  # 仍幂等

    async with session_factory() as s:
        convs = (await s.execute(select(Conversation))).scalars().all()
        msgs = (await s.execute(select(Message))).scalars().all()
    assert len(convs) == 1 + len(DEMO_CONVERSATIONS)
    assert len(msgs) == 1 + sum(len(turns) for turns in DEMO_CONVERSATIONS)
