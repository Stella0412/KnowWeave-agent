from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from app.config import Settings
from app.db.models import FlywheelSuggestion, KnowledgeChunk, LowConfidenceQuestion, LowConfidenceStatus
from app.services.flywheel_service import FlywheelService
from app.services.flywheel_suggestions import SuggestionPipeline, SuggestionResult


async def seed(factory):
    service = FlywheelService(factory)
    async with factory() as session:
        event = await service.record_event(session, conversation_id=None,
            raw_question='废砂盒多久清理一次', source='user_feedback', reason='缺少知识')
        await session.commit()
    return service, event.root_id


async def test_one_call_persists_all_three_fields_and_prefills_api(session_factory):
    service, root_id = await seed(session_factory)
    result = SuggestionResult(normalized_question='智能猫砂盆的废砂盒多久清理一次？',
        suggested_answer='请定期检查废砂盒装载情况，及时清理；具体周期请核实产品说明。',
        matched_chunk_id=None, match_reason='候选知识暂无对应清理周期，需要审核确认。')
    model = AsyncMock()
    model.ainvoke.return_value = result
    pipeline = SuggestionPipeline(session_factory, Settings(), model=model)
    await pipeline.generate(root_id)
    await pipeline.generate(root_id)
    assert model.ainvoke.await_count == 1
    items, _ = await service.list_queue()
    assert items[0]['normalized_question'] == result.normalized_question
    assert items[0]['suggested_answer'] == result.suggested_answer
    detail = await service.cluster_detail(root_id)
    assert detail['suggested_answer'] == result.suggested_answer
    assert detail['match_reason'] == result.match_reason
    assert detail['status'] == '待处理'
    async with session_factory() as session:
        assert not (await session.execute(select(KnowledgeChunk))).scalars().all()


async def test_rejection_during_generation_cannot_resurrect(session_factory):
    service, root_id = await seed(session_factory)
    model = AsyncMock()
    async def generate(_):
        await service.ignore(root_id, '频次：暂不沉淀')
        return SuggestionResult(normalized_question='废砂盒清理周期',
            suggested_answer='请核实产品说明。', match_reason='无匹配')
    model.ainvoke.side_effect = generate
    pipeline = SuggestionPipeline(session_factory, Settings(), model=model)
    assert await pipeline.generate(root_id) is None
    async with session_factory() as session:
        event = await service.record_event(session, conversation_id=None,
            raw_question='废砂盒多久清理一次', source='user_feedback', reason='再次提问')
        await session.commit()
        assert await session.get(FlywheelSuggestion, root_id) is None
        root = await session.get(LowConfidenceQuestion, root_id)
        assert root.status == LowConfidenceStatus.IGNORED
        assert root.duplicate_count == 2
    assert event.root_id == root_id
    assert (await service.list_queue())[1] == 0


async def test_failure_leaves_question_pending_and_retryable(session_factory):
    service, root_id = await seed(session_factory)
    model = AsyncMock()
    model.ainvoke.side_effect = RuntimeError('upstream offline')
    pipeline = SuggestionPipeline(session_factory, Settings(), model=model)
    with pytest.raises(RuntimeError):
        await pipeline.generate(root_id)
    async with session_factory() as session:
        assert await session.get(FlywheelSuggestion, root_id) is None
    assert (await service.cluster_detail(root_id))['status'] == '待处理'


async def test_invalid_dedup_target_is_not_saved(session_factory):
    _, root_id = await seed(session_factory)
    model = AsyncMock()
    model.ainvoke.return_value = SuggestionResult(normalized_question='清理周期',
        suggested_answer='核实说明后清理。', matched_chunk_id=999999, match_reason='匹配')
    with pytest.raises(ValueError):
        await SuggestionPipeline(session_factory, Settings(), model=model).generate(root_id)
    async with session_factory() as session:
        assert await session.get(FlywheelSuggestion, root_id) is None
