from datetime import datetime, timedelta

from app.db.models import LowConfidenceQuestion, LowConfidenceSource, LowConfidenceStatus
from app.db.repository import LowConfidenceRepository, TopicRepository

NAMES = ("退换货", "物流", "尺码", "其他")


async def _seed(session, rows):
    repo = LowConfidenceRepository(session)
    out = []
    for text, dup in rows:
        out.append(await repo.add_question(
            conversation_id=None, raw_question=text,
            source=LowConfidenceSource.RETRIEVAL_LOW_CONF, reason="test",
            message_id=None, fingerprint=text, merged_into_id=None,
            duplicate_count=dup, status=LowConfidenceStatus.PENDING,
            citations=None, knowledge_chunk_id=None, resolution=None,
        ))
    return out


async def test_upsert_and_distribution_weighted(session_factory):
    async with session_factory() as session:
        roots = await _seed(session, [("买大了想退", 3), ("物流到哪了", 2), ("发票怎么开", 1)])
        repo = TopicRepository(session)
        await repo.upsert_many([(roots[0].id, ["尺码", "退换货"]), (roots[1].id, ["物流"])])
        await repo.upsert_many([(roots[1].id, ["物流"])])
        await session.commit()

        result = await repo.distribution(names=NAMES)
        assert result["total_roots"] == 2
        assert result["total_weighted"] == 5
        by_label = {item["label"]: item["count"] for item in result["labels"]}
        assert by_label["物流"] == 1
        assert by_label["尺码"] == 1
        assert by_label["其他"] == 0
        weighted = await repo.distribution(names=NAMES, weighted=True)
        wb = {item["label"]: item["count"] for item in weighted["labels"]}
        assert wb["尺码"] == 3 and wb["物流"] == 2


async def test_list_unclassified_roots_excludes_classified(session_factory):
    async with session_factory() as session:
        roots = await _seed(session, [("a", 1), ("b", 1), ("c", 1)])
        repo = TopicRepository(session)
        await repo.upsert_many([(roots[0].id, ["其他"])])
        await session.commit()
        pending = await repo.list_unclassified_roots()
        assert [row.id for row in pending] == [roots[1].id, roots[2].id]


async def test_questions_by_label_and_coverage(session_factory):
    async with session_factory() as session:
        roots = await _seed(session, [("买大了想退", 4), ("物流到哪了", 1)])
        repo = TopicRepository(session)
        await repo.upsert_many([(roots[0].id, ["尺码", "退换货"])])
        await session.commit()
        items = await repo.questions_by_label("退换货")
        assert len(items) == 1
        assert items[0]["question_id"] == roots[0].id
        assert items[0]["duplicate_count"] == 4
        cov = await repo.coverage()
        assert cov["total_roots"] == 2 and cov["classified_roots"] == 1


async def test_distribution_since_filters_and_fills_all_labels(session_factory):
    async with session_factory() as session:
        roots = await _seed(session, [("x", 1)])
        repo = TopicRepository(session)
        await repo.upsert_many([(roots[0].id, ["其他"])])
        await session.commit()
        future = datetime.now() + timedelta(days=1)
        result = await repo.distribution(names=NAMES, since=future)
        assert result["total_roots"] == 0
        assert len(result["labels"]) == len(NAMES)
        assert all(item["count"] == 0 for item in result["labels"])
