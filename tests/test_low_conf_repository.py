from app.db.models import Conversation, LowConfidenceSource
from app.db.repository import LowConfidenceRepository


async def test_add_pool_entry_with_source_and_reason(session_factory):
    async with session_factory() as session:
        conv = Conversation(user_id="u1")
        session.add(conv)
        await session.flush()
        cid = conv.id
        await LowConfidenceRepository(session).add(
            cid, "退款几天到账", LowConfidenceSource.SELF_CHECK, "自评判证据不足以给出到账时间"
        )
        await session.commit()
    async with session_factory() as session:
        rows = await LowConfidenceRepository(session).list_questions()
    assert len(rows) == 1
    assert rows[0].raw_question == "退款几天到账"
    assert rows[0].source is LowConfidenceSource.SELF_CHECK
    assert "自评" in rows[0].reason
    assert rows[0].conversation_id == cid


async def test_add_pool_entry_without_conversation_is_allowed(session_factory):
    """conversation_id 在 DDL 里可空：离线评估跑出的拒答没有会话。"""
    async with session_factory() as session:
        await LowConfidenceRepository(session).add(
            None, "离线评估的问题", LowConfidenceSource.RETRIEVAL_LOW_CONF, "召回为空"
        )
        await session.commit()
    async with session_factory() as session:
        rows = await LowConfidenceRepository(session).list_questions()
    assert rows[0].conversation_id is None


async def test_list_filters_by_source_and_respects_limit(session_factory):
    async with session_factory() as session:
        repo = LowConfidenceRepository(session)
        for i in range(3):
            await repo.add(None, f"问题{i}", LowConfidenceSource.RETRIEVAL_LOW_CONF, "低分")
        await repo.add(None, "自评不过", LowConfidenceSource.SELF_CHECK, "自评")
        await session.commit()
    async with session_factory() as session:
        repo = LowConfidenceRepository(session)
        only_self = await repo.list_questions(source="self_check")
        limited = await repo.list_questions(limit=2)
    assert [r.raw_question for r in only_self] == ["自评不过"]
    assert len(limited) == 2


async def test_count_by_source_groups(session_factory):
    async with session_factory() as session:
        repo = LowConfidenceRepository(session)
        await repo.add(None, "a", LowConfidenceSource.RETRIEVAL_LOW_CONF, "x")
        await repo.add(None, "b", LowConfidenceSource.RETRIEVAL_LOW_CONF, "x")
        await repo.add(None, "c", LowConfidenceSource.SELF_CHECK, "x")
        await session.commit()
    async with session_factory() as session:
        counts = await LowConfidenceRepository(session).count_by_source()
    assert counts == {"retrieval_low_conf": 2, "self_check": 1}
