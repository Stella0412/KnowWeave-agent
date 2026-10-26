from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db.models import LowConfidenceSource, LowConfidenceStatus
from app.db.repository import LowConfidenceRepository, TopicRepository
from app.routers import topics as topics_router
from app.services.topic_admin_service import TopicAdminService


async def _seed(session_factory):
    async with session_factory() as session:
        repo = LowConfidenceRepository(session)
        a = await repo.add_question(
            conversation_id=None, raw_question="买大了想退",
            source=LowConfidenceSource.RETRIEVAL_LOW_CONF, reason="t",
            message_id=None, fingerprint="a", merged_into_id=None, duplicate_count=3,
            status=LowConfidenceStatus.PENDING, citations=None,
            knowledge_chunk_id=None, resolution=None,
        )
        await TopicRepository(session).upsert_many([(a.id, ["尺码", "退换货"])])
        await session.commit()


async def test_distribution_and_questions(session_factory):
    await _seed(session_factory)
    service = TopicAdminService(session_factory)
    dist = await service.distribution(days=30)
    assert dist["total_roots"] == 1
    assert len(dist["labels"]) == 17
    assert {i["label"]: i["count"] for i in dist["labels"]}["退换货"] == 1
    items = await service.questions("退换货")
    assert items[0]["raw_question"] == "买大了想退"


def test_api_routes_with_minimal_app(session_factory):
    mini = FastAPI()
    mini.state.topic_admin = TopicAdminService(session_factory)
    mini.include_router(topics_router.router)
    client = TestClient(mini)
    assert client.get("/topics/api/distribution").status_code == 200
    assert client.get("/topics/api/questions?label=退换货").status_code == 200
    assert client.get("/topics/api/questions?label=不存在").status_code == 400


def test_router_is_mounted_in_main_app():
    from app.main import app
    # FastAPI's lazy _IncludedRouter has no `.path`; its prefix lives on the original router.
    paths = {getattr(route, "path", "") for route in app.routes}
    prefixes = {
        getattr(getattr(route, "original_router", None), "prefix", "")
        for route in app.routes
    }
    assert "/topics" in paths or "/topics" in prefixes
