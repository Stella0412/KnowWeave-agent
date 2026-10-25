from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.db.models import (
    KnowledgeChunk, LowConfidenceQuestion, LowConfidenceStatus, QAExtractionStaging,
    StagingStatus, VectorizeStatus,
)
from app.db.repository import KnowledgeRepository, LowConfidenceRepository
from app.kb.chunking import Chunk
from app.routers import review as review_router
from app.services.flywheel_service import FlywheelService


class FakeMilvus:
    def __init__(self, boom=False):
        self.rows = []
        self.boom = boom

    def upsert(self, rows):
        if self.boom:
            raise RuntimeError("milvus down")
        self.rows.extend(rows)


def _embed(texts):
    return [[1.0, 0.0] for _ in texts]


async def _seed_root(session_factory, *, status=LowConfidenceStatus.PENDING):
    async with session_factory() as session:
        row = await LowConfidenceRepository(session).add_question(
            conversation_id=1, raw_question="保价规则是什么", source="user_feedback",
            reason="用户点踩", message_id=10, fingerprint="fp",
            merged_into_id=None, duplicate_count=2, status=status,
            citations=[{"n": 1, "chunk_id": 9, "answer": "旧片段"}],
            knowledge_chunk_id=None, resolution=None,
        )
        child = await LowConfidenceRepository(session).add_question(
            conversation_id=2, raw_question="价保怎么申请", source="retrieval_low_conf",
            reason="低分", message_id=None, fingerprint="fp",
            merged_into_id=row.id, duplicate_count=1,
            status=LowConfidenceStatus.MERGED, citations=[],
            knowledge_chunk_id=None, resolution=None,
        )
        await session.commit()
        return row.id, child.id


def _app(service):
    app = FastAPI()
    app.include_router(review_router.router)
    app.state.flywheel = service
    return app


async def test_queue_detail_metrics_and_comment(session_factory):
    root_id, child_id = await _seed_root(session_factory)
    service = FlywheelService(session_factory, embed=_embed, milvus_factory=FakeMilvus)
    app = _app(service)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        queue = await client.get("/review/api/queue?status=待处理")
        detail = await client.get(f"/review/api/clusters/{root_id}")
        metrics = await client.get("/review/api/metrics")
    assert queue.json()["total"] == 1
    assert queue.json()["status"] == "待处理"
    assert queue.json()["limit"] == 50
    assert queue.json()["offset"] == 0
    assert queue.json()["items"][0]["duplicate_count"] == 2
    assert queue.json()["items"][0]["sources"]["user_feedback"] == 1
    assert queue.json()["items"][0]["knowledge_chunk_id"] is None
    assert queue.json()["items"][0]["resolution"] is None
    body = detail.json()
    assert body["redirected_from"] is None
    assert body["variants"] == ["保价规则是什么", "价保怎么申请"]
    assert body["evidence_state"] == "has_snippets"
    assert len(body["occurrences"]) == 2
    assert body["occurrences"][0]["raw_question"] == "保价规则是什么"
    assert body["occurrences"][0]["created_at"] is not None
    assert metrics.json()["total_signals"] == 2   # 只累加根行的 duplicate_count
    assert metrics.json()["root_total"] == 1
    assert metrics.json()["status_counts"] == {"待处理": 1, "已归并": 1}
    assert metrics.json()["source_counts"] == {"user_feedback": 1, "retrieval_low_conf": 1}


async def test_child_redirect_and_orphan_detail_and_mutation(session_factory):
    root_id, child_id = await _seed_root(session_factory)
    service = FlywheelService(session_factory, embed=_embed, milvus_factory=FakeMilvus)
    app = _app(service)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ok = await client.get(f"/review/api/clusters/{child_id}")
        mutate_child = await client.post(
            f"/review/api/clusters/{child_id}/ignore", json={"resolution": "x"}
        )
        async with session_factory() as session:
            orphan = await LowConfidenceRepository(session).add_question(
                conversation_id=3, raw_question="孤儿", source="self_check",
                reason="r", message_id=None, fingerprint="orphan", merged_into_id=999,
                duplicate_count=1, status=LowConfidenceStatus.MERGED,
                citations=None, knowledge_chunk_id=None, resolution=None,
            )
            await session.commit()
            orphan_id = orphan.id
        missing = await client.get(f"/review/api/clusters/{orphan_id}")
        mutate = await client.post(
            f"/review/api/clusters/{orphan_id}/ignore", json={"resolution": "x"}
        )
    assert ok.json()["redirected_from"] == child_id
    assert ok.json()["id"] == root_id
    assert mutate_child.status_code == 409
    assert mutate_child.json()["root_id"] == root_id
    assert missing.status_code == 404
    assert missing.json()["orphan_id"] == orphan_id
    assert mutate.status_code == 409


async def test_answer_creates_kept_staging_pending_chunk_and_closes_root(session_factory):
    root_id, child_id = await _seed_root(session_factory)
    milvus = FakeMilvus()
    service = FlywheelService(session_factory, embed=_embed, milvus_factory=lambda: milvus)
    app = _app(service)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(f"/review/api/clusters/{root_id}/answer", json={
            "question": "保价规则是什么", "answer": "下单 7 天内可申请价保。",
            "category": "人工补充",
        })
    assert resp.status_code == 200
    assert resp.json()["vectorized"] is True
    async with session_factory() as session:
        root = await session.get(LowConfidenceQuestion, root_id)
        child = await session.get(LowConfidenceQuestion, child_id)
        staging = (await session.execute(select(QAExtractionStaging))).scalars().all()
        chunk = await session.get(KnowledgeChunk, root.knowledge_chunk_id)
    assert root.status is LowConfidenceStatus.CLOSED
    assert root.resolution.startswith("人工补充入库")
    assert child.status is LowConfidenceStatus.MERGED
    assert staging[0].status is StagingStatus.KEPT
    assert chunk.vectorize_status is VectorizeStatus.DONE


async def test_answer_reuses_existing_qa_and_closes_root(session_factory):
    root_id, _ = await _seed_root(session_factory)
    async with session_factory() as session:
        chunk = KnowledgeChunk(
            category="人工补充", questions="保价规则是什么",
            answer="下单 7 天内可申请价保。", content_type="gap_review",
            section_path="人工补充 > 飞轮#1", vectorize_status=VectorizeStatus.DONE,
        )
        session.add(chunk)
        await session.commit()
        chunk_id = chunk.id
    service = FlywheelService(session_factory, embed=_embed, milvus_factory=FakeMilvus)
    app = _app(service)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(f"/review/api/clusters/{root_id}/answer", json={
            "question": "保价规则是什么", "answer": "下单 7 天内可申请价保。",
        })
    assert resp.json()["reused"] is True
    assert resp.json()["knowledge_chunk_id"] == chunk_id
    async with session_factory() as session:
        root = await session.get(LowConfidenceQuestion, root_id)
        count = len((await session.execute(select(KnowledgeChunk))).scalars().all())
    assert root.status is LowConfidenceStatus.CLOSED
    assert root.knowledge_chunk_id == chunk_id
    assert count == 1


async def test_answer_vectorize_failure_keeps_pending(session_factory):
    root_id, _ = await _seed_root(session_factory)
    service = FlywheelService(
        session_factory, embed=_embed, milvus_factory=lambda: FakeMilvus(boom=True)
    )
    app = _app(service)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(f"/review/api/clusters/{root_id}/answer", json={
            "question": "保价规则是什么", "answer": "下单 7 天内可申请价保。",
        })
    assert resp.status_code == 202
    assert resp.json()["vectorize_status"] == "pending"
    async with session_factory() as session:
        root = await session.get(LowConfidenceQuestion, root_id)
        chunk = await session.get(KnowledgeChunk, root.knowledge_chunk_id)
    assert chunk.vectorize_status is VectorizeStatus.PENDING


async def test_reopen_is_only_way_back(session_factory):
    root_id, _ = await _seed_root(session_factory)
    service = FlywheelService(session_factory, embed=_embed, milvus_factory=FakeMilvus)
    app = _app(service)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ignored = await client.post(f"/review/api/clusters/{root_id}/ignore", json={"resolution": "不需要"})
        queue_after_ignore = await client.get("/review/api/queue?status=待处理")
        reopened = await client.post(f"/review/api/clusters/{root_id}/reopen", json={"resolution": "再审核"})
        queue_after_reopen = await client.get("/review/api/queue?status=待处理")
    assert ignored.json()["status"] == "已忽略"
    assert queue_after_ignore.json()["total"] == 0
    assert reopened.json()["status"] == "待处理"
    assert queue_after_reopen.json()["total"] == 1


async def test_vectorize_lock_conflict(session_factory):
    service = FlywheelService(session_factory, embed=_embed, milvus_factory=FakeMilvus)
    app = _app(service)
    transport = ASGITransport(app=app)
    await service._vectorize_lock.acquire()
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/review/api/vectorize")
    finally:
        service._vectorize_lock.release()
    assert resp.status_code == 409


async def test_vectorize_returns_pending_remaining(session_factory):
    async with session_factory() as session:
        await KnowledgeRepository(session).insert_chunks([
            Chunk(category="人工补充", questions="q", answer="a"),
        ])
        await session.commit()
    service = FlywheelService(session_factory, embed=_embed, milvus_factory=FakeMilvus)
    app = _app(service)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/review/api/vectorize")
    assert resp.status_code == 200
    assert resp.json()["vectorized"] == 1
    assert resp.json()["pending_remaining"] == 0
