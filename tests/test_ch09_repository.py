from app.db.models import (
    AnswerRating, KnowledgeChunk, LowConfidenceQuestion, LowConfidenceStatus,
    StagingStatus, VectorizeStatus,
)
from app.db.repository import (
    FeedbackRepository, KnowledgeRepository, LowConfidenceRepository, MetricsRepository,
)


async def test_feedback_roundtrip_and_comment_map(session_factory):
    async with session_factory() as session:
        repo = FeedbackRepository(session)
        row = await repo.add(1, 10, "down", "答非所问")
        await session.commit()
        assert await repo.get_by_message(10) is not None
        comments = await repo.comments_for_messages([10, 11])
    assert row.rating is AnswerRating.DOWN
    assert comments == {10: "答非所问"}


async def test_metrics_add(session_factory):
    async with session_factory() as session:
        await MetricsRepository(session).add(
            conversation_id=1, turn_token="t1", trace_id=None, route="business",
            intent="faq", status="ok", error_type=None, llm_calls=2,
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
            usage_estimated=False, model_usage={"m": {"calls": 1}},
            stages=None, cost_cny=0.1, latency_ms=100, first_token_ms=None,
            refusal_source=None, citations_count=1,
        )
        await session.commit()
    async with session_factory() as session:
        row = await session.get(__import__("app.db.models", fromlist=["TurnMetrics"]).TurnMetrics, 1)
    assert row.turn_token == "t1"


async def test_pool_root_and_child_queries(session_factory):
    async with session_factory() as session:
        repo = LowConfidenceRepository(session)
        root = await repo.add_question(
            conversation_id=1, raw_question="保价", source="user_feedback",
            reason="点踩", message_id=10, fingerprint="f", merged_into_id=None,
            duplicate_count=2, status=LowConfidenceStatus.CLOSED,
            citations=[], knowledge_chunk_id=None, resolution=None,
        )
        child = await repo.add_question(
            conversation_id=2, raw_question="保价", source="user_feedback",
            reason="点踩", message_id=11, fingerprint="f", merged_into_id=root.id,
            duplicate_count=1, status=LowConfidenceStatus.MERGED,
            citations=None, knowledge_chunk_id=None, resolution=None,
        )
        await session.commit()
        got = await repo.get_root_for_update("f")
        children = await repo.list_children(root.id)
        roots = await repo.list_roots(LowConfidenceStatus.CLOSED)
        total = await repo.count_roots(LowConfidenceStatus.CLOSED)
    assert got.id == root.id
    assert [c.id for c in children] == [child.id]
    assert [r.id for r in roots] == [root.id]
    assert total == 1


async def test_knowledge_pending_ids_and_qa_ids(session_factory):
    async with session_factory() as session:
        repo = KnowledgeRepository(session)
        ids = await repo.insert_chunks([
            __import__("app.kb.chunking", fromlist=["Chunk"]).Chunk(
                category="人工补充", questions="保价", answer="7 天价保",
            ),
        ])
        await session.commit()
        pending = await repo.list_pending(ids=ids)
        qa = await repo.list_existing_qa_with_ids()
        await repo.add_staging("b", "ref", "q", "a", status=StagingStatus.KEPT)
        await session.commit()
    assert [r.id for r in pending] == ids
    assert qa[0][0] == ids[0]
    assert qa[0][3] is VectorizeStatus.PENDING
