from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.db.repository import ChatRepository
from app.services.chat_service import ChatService


class FakeGraph:
    def __init__(self, values=None, messages=None):
        self.values = values or {}
        self.messages = messages or [AIMessage(content="答")]
        self.updated = []

    async def aget_state(self, config):
        return SimpleNamespace(values=self.values, tasks=[])

    async def aupdate_state(self, config, values, as_node=None):
        self.updated.append((config, values, as_node))
        self.values.update(values)

    async def astream(self, graph_input, config=None, context=None, stream_mode=None):
        self.values = {
            "messages": [HumanMessage(content="问"), AIMessage(content="答")],
            "answer": "答",
            "route": "business",
            "intent": "faq",
            "citations": [{"n": 1}],
            "suggested_actions": [],
        }
        yield ("updates", {"fallback_reply": {"answer": "答"}})

    async def ainvoke(self, graph_input, config=None, context=None):
        return {
            "messages": [HumanMessage(content="问"), AIMessage(content="答")],
            "answer": "答", "route": "business", "intent": "faq",
            "citations": [{"n": 1}], "suggested_actions": [],
        }


async def _service(session_factory, graph):
    return ChatService(
        model=object(), intent_model=object(), session_factory=session_factory,
        graph=graph,
    )


async def test_persist_new_returns_last_assistant_id(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    graph = FakeGraph()
    service = await _service(session_factory, graph)
    state = {"messages": [
        HumanMessage(content="问"), AIMessage(content="答一"),
        AIMessage(content=" "), AIMessage(content="答二"),
    ]}
    message_id = await service._persist_new(cid, state, 0)
    assert isinstance(message_id, int)
    async with session_factory() as session:
        rows = await ChatRepository(session).list_ui_messages(cid)
    assert [r.content for r in rows] == ["答一", "答二"]
    assert rows[-1].id == message_id


async def test_snapshot_write_caps_and_keeps_last_ten(session_factory):
    graph = FakeGraph(values={"citations_by_message_id": {"1": [{"n": 1}]}})
    service = await _service(session_factory, graph)
    state = {"citations": [{"n": 2, "answer": "x" * 5000}]}
    turn = SimpleNamespace(citations=[])
    for i in range(2, 14):
        await service._record_turn_snapshot(
            {"configurable": {"thread_id": "1"}}, state, turn, i
        )
    snapshots = graph.updated[-1][1]["citations_by_message_id"]
    assert list(snapshots)[-1] == "13"
    assert len(snapshots) == 10
    assert all(len(item["answer"]) <= 2000 for item in snapshots["13"])


async def test_citations_snapshot_survives_failed_turn(session_factory):
    # 上一轮失败会让新轮 citations 被清空；旧助手消息仍是最新，必须按 message_id 取旧快照。
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    graph = FakeGraph(values={"citations_by_message_id": {}})
    service = await _service(session_factory, graph)
    async with session_factory() as session:
        user = await ChatRepository(session).add_message(cid, role="user", content="问")
        msg = await ChatRepository(session).add_message(cid, role="assistant", content="答")
        await session.commit()
        message_id = msg.id
    graph.values["citations"] = []          # 失败后的新轮状态
    graph.values["citations_by_message_id"] = {
        str(message_id): [{"chunk_id": 7, "answer": "旧片段"}],
    }
    latest, snapshot = await service.latest_turn_snapshot(cid)
    assert latest == message_id
    assert snapshot == [{"chunk_id": 7, "answer": "旧片段"}]


async def test_empty_snapshot_is_distinct_from_missing_snapshot(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    graph = FakeGraph(values={"citations_by_message_id": {}})
    service = await _service(session_factory, graph)
    async with session_factory() as session:
        msg = await ChatRepository(session).add_message(cid, role="assistant", content="答")
        await session.commit()
        message_id = msg.id
    graph.values["citations_by_message_id"] = {str(message_id): []}
    latest, snapshot = await service.latest_turn_snapshot(cid)
    assert latest == message_id
    assert snapshot == []                   # 检索过但没片段
    graph.values["citations_by_message_id"] = {}
    latest, snapshot = await service.latest_turn_snapshot(cid)
    assert latest == message_id
    assert snapshot is None                 # 没有可用快照


async def test_stream_done_contains_message_id(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    graph = FakeGraph()
    service = await _service(session_factory, graph)
    events = [e async for e in service.stream(str(cid), "问")]
    assert events[-1]["type"] == "done"
    assert isinstance(events[-1]["message_id"], int)
