import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
from app.db.repository import ChatRepository
from app.prompts.context import SummaryOutput
from app.services.summary_service import SummaryRange, SummaryService
from app.services.token_counter import TokenCounter


class Snapshot:
    def __init__(self, messages):
        self.values = {"messages": messages}


class FakeGraph:
    def __init__(self, messages):
        self.messages = messages

    async def aget_state(self, config):
        return Snapshot(self.messages)


class FakeModel:
    def __init__(self, result=None, *, error=None, delay=0.0):
        self.result = result or SummaryOutput(has_facts=True, summary="用户要退订单1001的猫爬架。")
        self.error = error
        self.delay = delay
        self.seen = []

    async def ainvoke(self, messages):
        self.seen.append(messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


def _service(session_factory, graph, model):
    s = Settings(_env_file=None)
    return SummaryService(
        session_factory, model, graph, TokenCounter(estimate_factor=1.0), s
    )


def _messages():
    return [
        HumanMessage(content="我要退订单1001的猫爬架", id="db:1"),
        AIMessage(content="订单1001可以退，请填写表单。"),
        HumanMessage(content="表单在哪", id="db:2"),
        AIMessage(content="在回复下方。"),
    ]


async def _conversation(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    return cid


@pytest.mark.asyncio
async def test_summary_appends_segment_and_advances_anchor(session_factory):
    cid = await _conversation(session_factory)
    service = _service(session_factory, FakeGraph(_messages()), FakeModel())
    ok = await service.schedule(cid, SummaryRange(1, 2, None))
    assert ok is True
    await service.wait_for_idle()
    async with session_factory() as session:
        repo = ChatRepository(session)
        conv = await repo.get_conversation(cid)
        rows = await repo.list_summaries(cid)
    assert conv.summary_upto_msg_id == 2
    assert conv.summary.startswith("【第1段】")
    assert [r.seq for r in rows] == [1]
    assert rows[0].from_msg_id == 1
    assert rows[0].upto_msg_id == 2


@pytest.mark.asyncio
async def test_summary_schedule_does_not_block_caller(session_factory):
    cid = await _conversation(session_factory)
    service = _service(session_factory, FakeGraph(_messages()), FakeModel(delay=0.2))
    loop = asyncio.get_running_loop()
    start = loop.time()
    await service.schedule(cid, SummaryRange(1, 2, None))
    assert loop.time() - start < 0.1
    await service.wait_for_idle()


@pytest.mark.asyncio
async def test_stale_task_does_not_advance(session_factory):
    cid = await _conversation(session_factory)
    async with session_factory() as session:
        repo = ChatRepository(session)
        await repo.add_summary(cid, seq=1, from_msg_id=1, upto_msg_id=2, content="旧摘要")
        conv = await repo.get_conversation(cid)
        conv.summary_upto_msg_id = 2
        await session.commit()
    service = _service(session_factory, FakeGraph(_messages()), FakeModel())
    await service.schedule(cid, SummaryRange(1, 2, None))
    await service.wait_for_idle()
    async with session_factory() as session:
        rows = await ChatRepository(session).list_summaries(cid)
    assert [r.seq for r in rows] == [1]


@pytest.mark.asyncio
async def test_no_facts_advances_without_row(session_factory):
    cid = await _conversation(session_factory)
    model = FakeModel(SummaryOutput(has_facts=False, summary=""))
    service = _service(session_factory, FakeGraph(_messages()), model)
    await service.schedule(cid, SummaryRange(1, 2, None))
    await service.wait_for_idle()
    async with session_factory() as session:
        repo = ChatRepository(session)
        conv = await repo.get_conversation(cid)
        rows = await repo.list_summaries(cid)
    assert conv.summary_upto_msg_id == 2
    assert rows == []


@pytest.mark.asyncio
async def test_failure_does_not_advance_and_sets_cooldown(session_factory):
    cid = await _conversation(session_factory)
    service = _service(
        session_factory, FakeGraph(_messages()), FakeModel(error=RuntimeError("boom"))
    )
    await service.schedule(cid, SummaryRange(1, 2, None))
    await service.wait_for_idle()
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
    assert conv.summary_upto_msg_id is None
    assert await service.schedule(cid, SummaryRange(1, 2, None)) is False


@pytest.mark.asyncio
async def test_in_flight_single_flight(session_factory):
    cid = await _conversation(session_factory)
    service = _service(session_factory, FakeGraph(_messages()), FakeModel(delay=0.2))
    assert await service.schedule(cid, SummaryRange(1, 2, None)) is True
    assert await service.schedule(cid, SummaryRange(1, 2, None)) is False
    await service.wait_for_idle()


@pytest.mark.asyncio
async def test_fallback_skips_tool_rows(session_factory):
    cid = await _conversation(session_factory)
    async with session_factory() as session:
        repo = ChatRepository(session)
        await repo.add_message(cid, role="user", content="用户问题")
        await repo.add_message(cid, role="assistant", content="客服回答")
        await repo.add_message(cid, role="tool", content="工具结果")
        await session.commit()
    model = FakeModel()
    service = _service(session_factory, FakeGraph([]), model)
    await service.schedule(cid, SummaryRange(1, 3, None))
    await service.wait_for_idle()
    transcript = model.seen[0][1].content
    assert "用户问题" in transcript
    assert "客服回答" in transcript
    assert "工具" not in transcript
