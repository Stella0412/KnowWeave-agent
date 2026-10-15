import pytest

from app.db.models import FaithCaseStatus
from app.db.repository import FaithCaseRepository


async def _seed(session_factory, eval_id="B03", **over):
    payload = {
        "eval_id": eval_id, "bucket": "B_model", "query": "摄像头版是哪个型号",
        "strategy": "hybrid_rerank", "answer": "是 MH-CAM1[1]",
        "reason": "证据里只有 MH-FD30，答案却写了 MH-CAM1",
        "citations": [{"n": 1, "chunk_id": 42, "section_path": "商品规格手册 > 全景看护摄像头(型号 MH-CAM1)"}],
        "judge_model": "deepseek-chat",
    }
    payload.update(over)
    async with session_factory() as session:
        row = await FaithCaseRepository(session).upsert_case(**payload)
        await session.commit()
        return row.id


async def test_new_case_starts_open_with_seen_count_one(session_factory):
    await _seed(session_factory)
    async with session_factory() as session:
        cases = await FaithCaseRepository(session).list_cases()
    assert len(cases) == 1
    assert cases[0].status is FaithCaseStatus.OPEN
    assert cases[0].seen_count == 1


async def test_repeat_does_not_add_row_but_bumps_seen_count(session_factory):
    await _seed(session_factory)
    await _seed(session_factory, answer="又编了一版[2]")
    async with session_factory() as session:
        repo = FaithCaseRepository(session)
        cases = await repo.list_cases()
        await session.commit()
    assert len(cases) == 1
    assert cases[0].seen_count == 2
    assert cases[0].answer == "又编了一版[2]"      # 快照更新成最近一次
    assert cases[0].last_seen_at >= cases[0].first_seen_at


async def test_relapse_reverts_resolved_case_to_open(session_factory):
    """已解决的题又被判编造 → 那是复发，必须重新出现在待处理列表里（DDL 注释）。"""
    await _seed(session_factory)
    async with session_factory() as session:
        repo = FaithCaseRepository(session)
        await repo.set_status("B03", FaithCaseStatus.RESOLVED, resolution="补了 MH-CAM1 的规格条目")
        await session.commit()
    async with session_factory() as session:
        resolved = (await FaithCaseRepository(session).list_cases())[0]
        assert resolved.status is FaithCaseStatus.RESOLVED
        assert resolved.resolved_at is not None
        assert resolved.resolution == "补了 MH-CAM1 的规格条目"

    await _seed(session_factory)
    async with session_factory() as session:
        relapsed = (await FaithCaseRepository(session).list_cases())[0]
    assert relapsed.status is FaithCaseStatus.OPEN
    assert relapsed.seen_count == 2
    assert relapsed.resolved_at is not None        # 保留，用来标「复发」


async def test_set_status_back_to_open_clears_resolution(session_factory):
    await _seed(session_factory)
    async with session_factory() as session:
        repo = FaithCaseRepository(session)
        await repo.set_status("B03", FaithCaseStatus.NOT_NEEDED, resolution="该题标注错了")
        await repo.set_status("B03", FaithCaseStatus.OPEN)
        await session.commit()
    async with session_factory() as session:
        case = (await FaithCaseRepository(session).list_cases())[0]
    assert case.status is FaithCaseStatus.OPEN
    assert case.resolution is None      # 空着的处置在台账上等于没有交代 → 退回时清空


async def test_list_cases_can_filter_by_status(session_factory):
    await _seed(session_factory, eval_id="B03")
    await _seed(session_factory, eval_id="E01")
    async with session_factory() as session:
        repo = FaithCaseRepository(session)
        await repo.set_status("E01", FaithCaseStatus.RESOLVED, resolution="已补证据")
        await session.commit()
    async with session_factory() as session:
        open_cases = await FaithCaseRepository(session).list_cases(status="未解决")
    assert [c.eval_id for c in open_cases] == ["B03"]


async def test_set_status_unknown_id_returns_none(session_factory):
    async with session_factory() as session:
        assert await FaithCaseRepository(session).set_status("不存在", FaithCaseStatus.RESOLVED, "x") is None
