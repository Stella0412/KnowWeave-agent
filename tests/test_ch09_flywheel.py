import hashlib

import pytest
from sqlalchemy import select

from app.db.models import LowConfidenceQuestion, LowConfidenceStatus
from app.db.repository import LowConfidenceRepository
from app.rag.evidence import cap_citations
from app.services.flywheel_service import FlywheelService, question_fingerprint


async def _event(session_factory, **over):
    payload = dict(
        conversation_id=1, raw_question="保价规则是什么", source="user_feedback",
        reason="用户点踩", message_id=None, citations=None,
    )
    payload.update(over)
    service = FlywheelService(session_factory)
    async with session_factory() as session:
        result = await service.record_event(session, **payload)
        await session.commit()
    return result


def test_fingerprint_is_normalized_sha256():
    assert question_fingerprint(" 保价 规则？") == question_fingerprint("保价规则")
    assert question_fingerprint("保价规则") == hashlib.sha256(
        "保价规则".encode("utf-8")
    ).hexdigest()


async def test_new_then_hit_pending_increments_and_inserts_child(session_factory):
    first = await _event(session_factory, message_id=10, citations=[{"n": 1}])
    second = await _event(session_factory, message_id=11, citations=[])
    assert first.created is True and first.duplicate_count == 1
    assert second.created is False and second.root_id == first.root_id
    assert second.duplicate_count == 2
    async with session_factory() as session:
        rows = (await session.execute(select(LowConfidenceQuestion).order_by(
            LowConfidenceQuestion.id
        ))).scalars().all()
    assert [r.status for r in rows] == [
        LowConfidenceStatus.PENDING, LowConfidenceStatus.MERGED,
    ]
    assert rows[1].merged_into_id == rows[0].id
    assert rows[0].citations == [{"n": 1}]       # 命中不覆盖根快照
    assert rows[1].citations == []


@pytest.mark.parametrize("status", [LowConfidenceStatus.CLOSED, LowConfidenceStatus.IGNORED])
async def test_hit_closed_or_ignored_only_increments(session_factory, status):
    first = await _event(session_factory, message_id=20)
    async with session_factory() as session:
        row = await LowConfidenceRepository(session).get_for_update(first.root_id)
        row.status = status
        await session.commit()
    second = await _event(session_factory, message_id=21)
    assert second.duplicate_count == 2
    assert second.status == status.value
    async with session_factory() as session:
        row = await session.get(LowConfidenceQuestion, first.root_id)
    assert row.status is status
    assert row.duplicate_count == 2


async def test_pending_root_wins_when_same_fingerprint_has_multiple_statuses(session_factory):
    closed = await _event(session_factory, message_id=30)
    async with session_factory() as session:
        row = await LowConfidenceRepository(session).get_for_update(closed.root_id)
        row.status = LowConfidenceStatus.CLOSED
        await session.commit()
    # 模拟人工另开一个待处理根（同指纹、不同状态）
    async with session_factory() as session:
        pending = await LowConfidenceRepository(session).add_question(
            conversation_id=2, raw_question="保价规则是什么", source="user_feedback",
            reason="点踩", message_id=31, fingerprint=question_fingerprint("保价规则是什么"),
            merged_into_id=None, duplicate_count=1,
            status=LowConfidenceStatus.PENDING, citations=None,
            knowledge_chunk_id=None, resolution=None,
        )
        await session.commit()
    result = await _event(session_factory, message_id=32)
    assert result.root_id == pending.id
    assert result.duplicate_count == 2
    async with session_factory() as session:
        still_closed = await session.get(LowConfidenceQuestion, closed.root_id)
    assert still_closed.status is LowConfidenceStatus.CLOSED


async def test_orphan_child_creates_new_root_without_rewriting_orphan(session_factory):
    fp = question_fingerprint("售后电话是多少")
    async with session_factory() as session:
        orphan = await LowConfidenceRepository(session).add_question(
            conversation_id=1, raw_question="售后电话是多少", source="user_feedback",
            reason="点踩", message_id=40, fingerprint=fp, merged_into_id=9999,
            duplicate_count=1, status=LowConfidenceStatus.MERGED,
            citations=None, knowledge_chunk_id=None, resolution=None,
        )
        await session.commit()
        orphan_id = orphan.id
    result = await _event(session_factory, raw_question="售后电话是多少", message_id=41)
    assert result.created is True
    assert result.root_id != orphan_id
    async with session_factory() as session:
        orphan = await session.get(LowConfidenceQuestion, orphan_id)
        root = await session.get(LowConfidenceQuestion, result.root_id)
    assert orphan.merged_into_id == 9999
    assert orphan.duplicate_count == 1
    assert orphan.status is LowConfidenceStatus.MERGED
    assert root.merged_into_id is None
    assert root.duplicate_count == 1


async def test_empty_question_raises(session_factory):
    with pytest.raises(ValueError):
        await _event(session_factory, raw_question="   ")


def test_cap_citations_limits_items_answer_and_bytes():
    items = [{"n": i, "answer": "x" * 5000} for i in range(30)]
    capped = cap_citations(items)
    assert len(capped) <= 20
    assert all(len(item["answer"]) <= 2000 for item in capped)
    import json
    assert len(json.dumps(capped, ensure_ascii=False).encode("utf-8")) <= 65536
