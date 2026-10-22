import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import Settings
from app.db.repository import ChatRepository
from app.services.chat_service import ChatService


class Snapshot:
    def __init__(self, values):
        self.values = values
        self.tasks = []


class FakeGraph:
    def __init__(self, existing=None, final=None):
        self.existing = existing or []
        self.final = final or {}
        self.inputs = []

    async def aget_state(self, config):
        if self.inputs:
            return Snapshot(self.final)
        return Snapshot({"messages": list(self.existing)})

    async def astream(self, *args, **kwargs):
        if False:
            yield None

    async def ainvoke(self, input, config, context):
        self.inputs.append(input)
        return self.final


def _service(session_factory, graph, **overrides):
    s = Settings(_env_file=None, **overrides)
    return ChatService(
        model=object(), intent_model=object(), session_factory=session_factory,
        graph=graph, settings=s,
    )


@pytest.mark.asyncio
async def test_overlong_user_input_is_rejected_before_conversation_created(session_factory):
    service = _service(session_factory, FakeGraph(), max_user_input_tokens=5)
    with pytest.raises(ValueError, match="用户输入过长"):
        await service.run(None, "很长" * 100)
    async with session_factory() as session:
        rows = await ChatRepository(session).list_conversations_for_user()
    assert rows == []


@pytest.mark.asyncio
async def test_persist_skips_tool_rows_and_empty_assistant(session_factory):
    final = {
        "messages": [
            HumanMessage(content="问", id="db:1"),
            AIMessage(content="", tool_calls=[{
                "name": "query_order", "args": {"order_id": "1001"},
                "id": "c1", "type": "tool_call",
            }]),
            ToolMessage(content="订单结果", tool_call_id="c1"),
            AIMessage(content="答"),
        ],
        "answer": "答",
    }
    graph = FakeGraph(final=final)
    service = _service(session_factory, graph)
    payload = await service.run(None, "问")
    async with session_factory() as session:
        rows = await ChatRepository(session).list_messages(payload["session_id"])
    assert [r.role.value for r in rows] == ["user", "assistant"]
    assert rows[1].content == "答"


@pytest.mark.asyncio
async def test_seed_filters_tool_rows_and_strips_tool_calls(session_factory):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        await repo.add_message(cid, role="user", content="旧问")
        await repo.add_message(cid, role="assistant", content="旧答", tool_calls=[{
            "name": "query_order", "args": {},
            "id": "old", "type": "tool_call",
        }])
        await repo.add_message(cid, role="tool", content="旧工具", tool_call_id="old")
        await session.commit()
    final = {
        "messages": [
            HumanMessage(content="旧问", id="db:1"),
            AIMessage(content="旧答", id="db:2"),
            HumanMessage(content="新问", id="db:4"),
            AIMessage(content="新答"),
        ]
    }
    graph = FakeGraph(existing=[], final=final)
    await _service(session_factory, graph).run(str(cid), "新问")
    sent = graph.inputs[0]["messages"]
    assert [m.content for m in sent] == ["旧问", "旧答", "新问"]
    assert sent[1].tool_calls == []


@pytest.mark.asyncio
async def test_user_message_touches_conversation_updated_at(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
        before = conv.updated_at
    time.sleep(0.01)
    await _service(session_factory, FakeGraph()).run(str(cid), "新问题")
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
    assert conv.updated_at >= before
