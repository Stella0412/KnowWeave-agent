from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.db.models import AnswerFeedback, LowConfidenceQuestion
from app.db.repository import ChatRepository
from app.routers import feedback as feedback_router
from app.services.feedback_service import FeedbackRejected, FeedbackService
from app.services.flywheel_service import FlywheelService


class FakeChat:
    def __init__(self, latest, snapshot):
        self.latest = latest
        self.snapshot = snapshot

    async def latest_turn_snapshot(self, conversation_id):
        return self.latest, self.snapshot


async def _setup(session_factory, *, latest=True, snapshot=None):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        user = await repo.add_message(cid, role="user", content="保价规则是什么")
        answer = await repo.add_message(cid, role="assistant", content="不知道")
        await session.commit()
    latest_id = answer.id if latest else -1
    service = FeedbackService(
        session_factory, FakeChat(latest_id, snapshot), FlywheelService(session_factory)
    )
    return cid, user.id, answer.id, service


async def test_down_creates_feedback_and_pool_root(session_factory):
    cid, _, message_id, service = await _setup(
        session_factory, snapshot=[{"n": 1, "chunk_id": 9}]
    )
    result = await service.submit(
        conversation_id=cid, message_id=message_id, rating="down", comment="答非所问"
    )
    assert result["pooled"] is True
    assert result["occurrences"] == 1
    assert result["citations_saved"] is True
    async with session_factory() as session:
        feedback = await session.get(AnswerFeedback, result["feedback_id"])
        pool = await session.get(LowConfidenceQuestion, result["cluster_id"])
    assert feedback.comment == "答非所问"
    assert pool.message_id == message_id
    assert pool.reason == "用户点踩：答非所问"
    assert pool.citations == [{"n": 1, "chunk_id": 9}]


async def test_up_does_not_pool(session_factory):
    cid, _, message_id, service = await _setup(session_factory)
    result = await service.submit(
        conversation_id=cid, message_id=message_id, rating="up", comment=None
    )
    assert result["pooled"] is False
    assert result["cluster_id"] is None
    async with session_factory() as session:
        assert (await session.get(AnswerFeedback, result["feedback_id"])).rating.value == "up"


async def test_same_rating_is_idempotent_and_opposite_is_409(session_factory):
    cid, _, message_id, service = await _setup(session_factory, snapshot=[])
    first = await service.submit(
        conversation_id=cid, message_id=message_id, rating="down", comment=None
    )
    again = await service.submit(
        conversation_id=cid, message_id=message_id, rating="down", comment=None
    )
    assert again["duplicate"] is True
    assert again["cluster_id"] == first["cluster_id"]
    assert again["occurrences"] == 1
    with pytest.raises(FeedbackRejected) as exc:
        await service.submit(
            conversation_id=cid, message_id=message_id, rating="up", comment=None
        )
    assert exc.value.status_code == 409


async def test_feedback_rejects_non_latest(session_factory):
    cid, _, message_id, service = await _setup(session_factory, latest=False)
    with pytest.raises(FeedbackRejected) as exc:
        await service.submit(
            conversation_id=cid, message_id=message_id, rating="down", comment=None
        )
    assert exc.value.status_code == 409


async def test_feedback_missing_snapshot_is_stored_as_null(session_factory):
    cid, _, message_id, service = await _setup(session_factory, snapshot=None)
    result = await service.submit(
        conversation_id=cid, message_id=message_id, rating="down", comment=None
    )
    assert result["citations_saved"] is False
    async with session_factory() as session:
        pool = await session.get(LowConfidenceQuestion, result["cluster_id"])
    assert pool.citations is None


async def test_router_validation_and_status_codes(session_factory):
    cid, _, message_id, service = await _setup(session_factory, snapshot=[])
    app = FastAPI()
    app.include_router(feedback_router.router)
    app.state.feedback_service = service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ok = await client.post("/api/feedback", json={
            "conversation_id": cid, "message_id": message_id,
            "rating": "down", "comment": "  ",
        })
        invalid = await client.post("/api/feedback", json={
            "conversation_id": cid, "message_id": message_id,
            "rating": "down", "comment": "x" * 301,
        })
        extra = await client.post("/api/feedback", json={
            "conversation_id": cid, "message_id": message_id,
            "rating": "down", "unexpected": 1,
        })
    assert ok.status_code == 200
    assert ok.json()["pooled"] is True
    assert invalid.status_code == 422
    assert extra.status_code == 422
