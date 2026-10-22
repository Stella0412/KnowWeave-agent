import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
from app.db.repository import ChatRepository
from app.services.context_manager import ContextManager
from app.services.token_counter import ContextBudget, TokenCounter


class FakeSummary:
    def __init__(self):
        self.calls = []

    async def schedule(self, conversation_id, frozen):
        self.calls.append((conversation_id, frozen))
        return True


def _counter():
    return TokenCounter(estimate_factor=1.0)


def _budget(layer1=1000, layer2=1000):
    return ContextBudget(
        model_window=20000, fixed=1000, peak=1000,
        window_history=18000, history=layer1 + layer2,
        layer1=layer1, layer2=layer2,
    )


def _turn(anchor, text, answer):
    return [
        HumanMessage(content=text, id=f"db:{anchor}"),
        AIMessage(content=answer),
    ]


async def _conversation(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    return cid


def _manager(session_factory, budget, summary=None):
    return ContextManager(
        session_factory,
        counter=_counter(),
        budget=budget,
        settings=Settings(_env_file=None, classifier_history_max_tokens=80),
        summary_service=summary,
    )


@pytest.mark.asyncio
async def test_layer1_demotes_oldest_turn_and_persists_anchor(session_factory, caplog):
    cid = await _conversation(session_factory)
    messages = _turn(1, "旧问题" * 20, "旧回答" * 20) + _turn(2, "近问题", "近回答") + _turn(3, "当前问题", "")
    counter = _counter()
    second_turn_tokens = counter.count_messages(_turn(2, "近问题", "近回答"))
    manager = _manager(session_factory, _budget(layer1=second_turn_tokens))
    with caplog.at_level(logging.INFO, logger="knowweave.context"):
        meta = await manager.prepare(cid, messages, current_user_msg_id=3)
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
    assert conv.layer1_from_msg_id == 2
    assert meta["layer1_from"] == 2
    assert meta["demoted"] is True
    assert "layer1 降级" in caplog.text


@pytest.mark.asyncio
async def test_layer2_over_budget_schedules_frozen_summary(session_factory):
    cid = await _conversation(session_factory)
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
        conv.layer1_from_msg_id = 2
        await session.commit()
    messages = _turn(1, "旧问题" * 30, "旧回答" * 30) + _turn(2, "近问题", "近回答") + _turn(3, "当前问题", "")
    summary = FakeSummary()
    manager = _manager(session_factory, _budget(layer1=10000, layer2=1), summary)
    meta = await manager.prepare(cid, messages, current_user_msg_id=3)
    assert meta["summary_triggered"] is True
    assert len(summary.calls) == 1
    _, frozen = summary.calls[0]
    assert (frozen.from_msg_id, frozen.upto_msg_id) == (1, 1)


@pytest.mark.asyncio
async def test_effective_summary_uses_segments_and_selects_first_plus_recent(session_factory):
    cid = await _conversation(session_factory)
    async with session_factory() as session:
        repo = ChatRepository(session)
        await repo.add_summary(cid, seq=1, from_msg_id=1, upto_msg_id=1, content="最早订单1001")
        await repo.add_summary(cid, seq=2, from_msg_id=2, upto_msg_id=2, content="中间退货")
        await repo.add_summary(cid, seq=3, from_msg_id=3, upto_msg_id=3, content="最近补发")
        await session.commit()
    messages = _turn(1, "问1", "答1") + _turn(2, "问2", "答2") + _turn(3, "问3", "答3") + _turn(4, "当前", "")
    manager = _manager(session_factory, _budget(layer1=10000, layer2=10000))
    meta = await manager.prepare(cid, messages, current_user_msg_id=4)
    assert "最早订单1001" in meta["summary_text"]
    assert meta["summary_segments"]


@pytest.mark.asyncio
async def test_layer2_overflow_keeps_latest_turns_with_placeholder(session_factory):
    cid = await _conversation(session_factory)
    manager = _manager(session_factory, _budget(layer1=10000, layer2=30))
    messages = (
        _turn(1, "最早" * 20, "答1")
        + _turn(2, "中间" * 20, "答2")
        + _turn(3, "最近" * 20, "答3")
        + _turn(4, "当前", "")
    )
    meta = {
        "conversation_id": cid,
        "turn_user_msg_id": 4,
        "effective_summary_upto": None,
        "layer1_from": 4,
        "summary_text": "",
    }
    prefix, current, react = manager.model_history(messages, meta)
    assert prefix[0].content == "【层 2 早期内容待摘要】"
    assert any("最近" in str(m.content) for m in prefix)
    assert not any("最早" in str(m.content) for m in prefix)
    assert current.content == "当前"
    assert react == []


@pytest.mark.asyncio
async def test_history_ctx_logged_every_turn(session_factory, caplog):
    cid = await _conversation(session_factory)
    manager = _manager(session_factory, _budget())
    with caplog.at_level(logging.INFO, logger="knowweave.context"):
        await manager.prepare(cid, _turn(1, "问", "答") + _turn(2, "当前", ""), 2)
    assert "history_ctx" in caplog.text
    assert "window_count" in caplog.text


@pytest.mark.asyncio
async def test_shared_history_starts_on_human_and_respects_budget(session_factory):
    cid = await _conversation(session_factory)
    manager = _manager(session_factory, _budget(), FakeSummary())
    messages = _turn(1, "旧问题" * 30, "旧回答" * 30) + _turn(2, "当前问题", "")
    meta = await manager.prepare(cid, messages, current_user_msg_id=2)
    history = manager.shared_history(messages, meta)
    assert history
    assert isinstance(history[0], HumanMessage)
    prefix, current, react = manager.model_history(messages, meta)
    assert current.content == "当前问题"
    assert react == []
    assert all(not isinstance(m, HumanMessage) or m.content != "当前问题" for m in prefix)


@pytest.mark.asyncio
async def test_shared_history_fallback_caps_single_long_message(session_factory):
    cid = await _conversation(session_factory)
    counter = _counter()
    manager = _manager(session_factory, _budget())
    messages = _turn(1, "旧问题" * 30, "旧回答" * 30) + _turn(2, "当前问题", "")
    meta = await manager.prepare(cid, messages, current_user_msg_id=2)
    history = manager.shared_history(messages, meta)
    assert history
    assert isinstance(history[0], HumanMessage)
    assert counter.count_messages(history) <= 80


@pytest.mark.asyncio
async def test_prepare_persists_initialized_anchor_without_demotion(session_factory):
    cid = await _conversation(session_factory)
    manager = _manager(session_factory, _budget())
    meta = await manager.prepare(
        cid, _turn(1, "问", "答") + _turn(2, "当前", ""), current_user_msg_id=2
    )
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
    assert conv.layer1_from_msg_id == 1
    assert meta["layer1_from"] == 1
    assert meta["demoted"] is False
