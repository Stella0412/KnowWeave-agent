import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, func

from app.db.models import (FlywheelSuggestion, FlywheelSuggestionEvidence,
    FlywheelReviewDecision, KnowledgeChunk, LowConfidenceQuestion, LowConfidenceStatus, VectorizeStatus)
from app.db.repository import KnowledgeRepository
from app.kb.chunking import Chunk
from app.routers.review import router
from app.services.flywheel_service import FlywheelService, ReviewRejected


class FakeMilvus:
    def __init__(self, fail=False):
        self.rows = []
        self.fail = fail

    def upsert(self, rows):
        if self.fail:
            raise RuntimeError('offline')
        self.rows.extend(rows)


async def setup(factory, *, fail=False):
    milvus = FakeMilvus(fail)
    service = FlywheelService(factory, embed=lambda texts: [[1., 0.] for _ in texts],
                              milvus_factory=lambda: milvus)
    async with factory() as session:
        target = (await KnowledgeRepository(session).insert_chunks([
            Chunk(category='产品说明', questions='废砂盒如何清理？', answer='每日检查，按需清理。',
                  content_type='faq', section_path='产品手册 > 清洁', is_key_clause=False)
        ], status=VectorizeStatus.DONE))[0]
        event = await service.record_event(session, conversation_id=None,
            raw_question='废砂盒怎么清理', source='user_feedback', reason='未解答',
            citations=[{'chunk_id': target, 'answer': '历史引用原文'}])
        session.add(FlywheelSuggestion(question_id=event.root_id,
            normalized_question='废砂盒的清洁方式是什么？', suggested_answer='每日检查，及时倒掉废砂并清洁盒体。',
            matched_chunk_id=target, match_reason='同一清洁问题；请核实适用型号。'))
        session.add(FlywheelSuggestionEvidence(question_id=event.root_id,
            knowledge_candidates=[{'id': target, 'question': '废砂盒如何清理？', 'answer': '生成时的候选原文'}],
            retrieval_snapshot=[{'chunk_id': target, 'answer': '生成时检索原文'}]))
        await session.commit()
    detail = await service.cluster_detail(event.root_id)
    return service, event.root_id, target, detail['matched_knowledge']['signature'], milvus


async def test_reuse_links_target_without_overwriting_or_duplicating(session_factory):
    service, root, target, signature, milvus = await setup(session_factory)
    result = await service.submit_answer(root, question='忽略的输入', answer='不得覆盖原答案',
        action='reuse', target_chunk_id=target, expected_signature=signature)
    assert result['action'] == 'reuse' and result['vectorized']
    async with session_factory() as session:
        chunk = await session.get(KnowledgeChunk, target)
        assert chunk.answer == '每日检查，按需清理。'
        assert await session.scalar(select(func.count()).select_from(KnowledgeChunk)) == 1
        question = await session.get(LowConfidenceQuestion, root)
        assert question.knowledge_chunk_id == target and question.status == LowConfidenceStatus.CLOSED
        decision = (await session.execute(select(FlywheelReviewDecision))).scalar_one()
        assert decision.action == 'reuse' and decision.before == decision.after
    assert milvus.rows == []


@pytest.mark.parametrize('fail', [False, True])
async def test_update_reuses_id_audits_before_after_and_resynchronizes(session_factory, fail):
    service, root, target, signature, milvus = await setup(session_factory, fail=fail)
    result = await service.submit_answer(root, question='废砂盒如何清洁？', answer='核准后的清洁方法。',
        action='update', target_chunk_id=target, expected_signature=signature, category=None)
    assert result['action'] == 'update' and result['vectorized'] is (not fail)
    async with session_factory() as session:
        chunk = await session.get(KnowledgeChunk, target)
        assert chunk.answer == '核准后的清洁方法。' and chunk.category == '产品说明'
        assert chunk.vectorize_status == (VectorizeStatus.PENDING if fail else VectorizeStatus.DONE)
        assert await session.scalar(select(func.count()).select_from(KnowledgeChunk)) == 1
        decision = (await session.execute(select(FlywheelReviewDecision))).scalar_one()
        assert decision.before['answer'] == '每日检查，按需清理。'
        assert decision.after['answer'] == '核准后的清洁方法。'
    if not fail:
        assert milvus.rows[0]['id'] == target
        assert milvus.rows[0]['answer'] == '核准后的清洁方法。'


@pytest.mark.parametrize('mode', ['reuse', 'update'])
async def test_stale_version_rejected_without_closing_root(session_factory, mode):
    service, root, target, signature, _ = await setup(session_factory)
    async with session_factory() as session:
        chunk = await session.get(KnowledgeChunk, target)
        chunk.answer = '其他审核人修订的内容'
        await session.commit()
    with pytest.raises(ReviewRejected) as exc:
        await service.submit_answer(root, question='问题', answer='过期内容', action=mode,
            target_chunk_id=target, expected_signature=signature)
    assert exc.value.status_code == 409
    async with session_factory() as session:
        assert (await session.get(LowConfidenceQuestion, root)).status == LowConfidenceStatus.PENDING
        assert (await session.get(KnowledgeChunk, target)).answer == '其他审核人修订的内容'
        assert await session.scalar(select(func.count()).select_from(FlywheelReviewDecision)) == 0


async def test_arbitrary_target_and_missing_target_rejected(session_factory):
    service, root, target, signature, _ = await setup(session_factory)
    for bad_target in [None, target + 100]:
        with pytest.raises(ReviewRejected) as exc:
            await service.submit_answer(root, question='问题', answer='答案', action='reuse',
                target_chunk_id=bad_target, expected_signature=signature)
        assert exc.value.status_code in [422, 409]


async def test_explicit_new_creates_separate_knowledge_and_audit(session_factory):
    service, root, target, _, _ = await setup(session_factory)
    result = await service.submit_answer(root, question='不同型号的清洁方法', answer='核准的新知识。', action='new')
    assert result['action'] == 'new' and result['knowledge_chunk_id'] != target
    async with session_factory() as session:
        assert (await session.get(KnowledgeChunk, target)).answer == '每日检查，按需清理。'
        assert await session.scalar(select(func.count()).select_from(KnowledgeChunk)) == 2


async def test_detail_distinguishes_original_evidence_from_current_knowledge(session_factory):
    service, root, target, _, _ = await setup(session_factory)
    detail = await service.cluster_detail(root)
    assert detail['matched_knowledge']['answer'] == '每日检查，按需清理。'
    assert detail['matched_knowledge']['section_path'] == '产品手册 > 清洁'
    assert detail['suggestion_evidence']['knowledge_candidates'][0]['answer'] == '生成时的候选原文'
    assert detail['suggestion_evidence']['retrieval_snapshot'][0]['answer'] == '生成时检索原文'
    async with session_factory() as session:
        await session.delete(await session.get(FlywheelSuggestionEvidence, root))
        await session.commit()
    assert (await service.cluster_detail(root))['suggestion_evidence'] is None


async def test_api_reuse_accepts_no_draft_and_rejects_invalid_action(session_factory):
    service, root, target, signature, _ = await setup(session_factory)
    app = FastAPI()
    app.include_router(router)
    app.state.flywheel = service
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        bad = await client.post(f'/review/api/clusters/{root}/answer', json={'action': 'delete'})
        assert bad.status_code == 422
        response = await client.post(f'/review/api/clusters/{root}/answer', json={
            'action': 'reuse', 'target_chunk_id': target, 'expected_signature': signature})
    assert response.status_code == 200
    assert response.json()['action'] == 'reuse'
