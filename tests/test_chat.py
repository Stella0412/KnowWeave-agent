from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.db.models import Conversation
from app.db.repository import ChatRepository
from app.routers import chat as chat_module
from app.services.chat_service import ChatService


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeSnapshot:
    def __init__(self, values):
        self.values = values


class FakeGraph:
    def __init__(self, events, final_state, existing=None):
        self._events = events
        self._final_state = final_state
        self._existing = existing or []
        self.inputs = []

    async def aget_state(self, config):
        # 首次调用发生在 _prepare（看 checkpoint 是否为空）；
        # 流结束后再调用时返回最终状态，供 ChatService 落库和发 citations/actions。
        if self.inputs:
            return FakeSnapshot(self._final_state)
        return FakeSnapshot({"messages": list(self._existing)})

    async def astream(self, input, config, context, stream_mode):
        self.inputs.append(input)
        for event in self._events:
            yield event

    async def ainvoke(self, input, config, context):
        self.inputs.append(input)
        return self._final_state


async def _collect(service, session_id, message):
    events = []
    async for event in service.stream(session_id, message):
        events.append(event)
    return events


def _service(graph, session_factory, intent_model=None):
    return ChatService(
        model=object(),
        intent_model=intent_model or object(),
        session_factory=session_factory,
        graph=graph,
    )


@pytest.mark.asyncio
async def test_new_session_emits_session_and_persists(session_factory):
    final = {
        "messages": [
            HumanMessage(content="你好", id="db:1"),
            AIMessage(content="您好"),
        ],
        "citations": [],
        "suggested_actions": [],
        "answer": "您好",
    }
    graph = FakeGraph([
        ("messages", (FakeChunk("您"), {"langgraph_node": "agent_llm"})),
        ("messages", (FakeChunk("好"), {"langgraph_node": "agent_llm"})),
    ], final)
    events = await _collect(_service(graph, session_factory), None, "你好")
    assert events[0]["type"] == "session"
    assert events[-1]["type"] == "done"
    cid = int(events[0]["session_id"])
    async with session_factory() as session:
        rows = await ChatRepository(session).list_messages(cid)
    assert [r.role.value for r in rows] == ["user", "assistant"]
    assert rows[1].content == "您好"


@pytest.mark.asyncio
async def test_filters_classify_intent_tokens(session_factory):
    final = {"messages": [HumanMessage(content="你好", id="db:1"), AIMessage(content="好")]}
    graph = FakeGraph([
        ("messages", (FakeChunk("错误"), {"langgraph_node": "classify_intent"})),
        ("messages", (FakeChunk("好"), {"langgraph_node": "agent_llm"})),
    ], final)
    events = await _collect(_service(graph, session_factory), None, "你好")
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == "好"


@pytest.mark.asyncio
async def test_fixed_reply_streams_from_updates(session_factory):
    final = {"messages": [HumanMessage(content="你好", id="db:1"), AIMessage(content="您好呀")]}
    graph = FakeGraph([
        ("updates", {"chitchat_reply": {"answer": "您好呀"}}),
    ], final)
    events = await _collect(_service(graph, session_factory), None, "你好")
    assert [e["content"] for e in events if e["type"] == "token"] == ["您好呀"]


@pytest.mark.asyncio
async def test_tool_status_comes_from_agent_update(session_factory):
    tool_call = {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}
    final = {
        "messages": [
            HumanMessage(content="订单", id="db:1"),
            AIMessage(content="", tool_calls=[tool_call]),
            ToolMessage(content="订单信息", tool_call_id="c1"),
            AIMessage(content="答"),
        ],
        "citations": [],
        "suggested_actions": [],
        "answer": "答",
    }
    graph = FakeGraph([
        ("updates", {"agent_llm": {"messages": [AIMessage(content="", tool_calls=[tool_call])]}}),
    ], final)
    events = await _collect(_service(graph, session_factory), None, "订单")
    status = [e for e in events if e["type"] == "tool_status"]
    assert status == [{"type": "tool_status", "name": "query_order", "status": "running"}]


@pytest.mark.asyncio
async def test_citations_and_actions_before_done(session_factory):
    final = {
        "messages": [HumanMessage(content="投诉", id="db:1"), AIMessage(content="抱歉")],
        "citations": [{"n": 1, "chunk_id": 7}],
        "suggested_actions": [{"type": "transfer_human", "label": "转人工"}],
        "answer": "抱歉",
    }
    graph = FakeGraph([
        ("updates", {"complaint_reply": {"answer": "抱歉"}}),
    ], final)
    events = await _collect(_service(graph, session_factory), None, "投诉")
    types = [e["type"] for e in events]
    assert types.index("citations") < types.index("actions") < types.index("done")


@pytest.mark.asyncio
async def test_persists_tool_trace(session_factory):
    call = {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}
    final = {
        "messages": [
            HumanMessage(content="订单", id="db:1"),
            AIMessage(content="", tool_calls=[call]),
            ToolMessage(content="订单信息", tool_call_id="c1"),
            AIMessage(content="答"),
        ],
        "citations": [],
        "suggested_actions": [],
        "answer": "答",
    }
    graph = FakeGraph([], final)
    events = await _collect(_service(graph, session_factory), None, "订单")
    cid = int(events[0]["session_id"])
    async with session_factory() as session:
        rows = await ChatRepository(session).list_messages(cid)
    assert [r.role.value for r in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[1].tool_calls[0]["name"] == "query_order"
    assert rows[2].tool_call_id == "c1"


@pytest.mark.asyncio
async def test_seeds_mysql_history_when_checkpoint_empty(session_factory):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        await repo.add_message(cid, role="user", content="旧问题")
        await repo.add_message(cid, role="assistant", content="旧回答")
        await session.commit()

    final = {
        "messages": [
            HumanMessage(content="旧问题", id="db:1"),
            AIMessage(content="旧回答", id="db:2"),
            HumanMessage(content="新问题", id="db:3"),
            AIMessage(content="新回答"),
        ]
    }
    graph = FakeGraph([], final, existing=[])
    await _collect(_service(graph, session_factory), str(cid), "新问题")
    sent = graph.inputs[0]["messages"]
    assert [m.content for m in sent] == ["旧问题", "旧回答", "新问题"]


@pytest.mark.asyncio
async def test_second_turn_does_not_reseed_when_checkpoint_has_messages(session_factory):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        await repo.add_message(cid, role="user", content="旧问题")
        await repo.add_message(cid, role="assistant", content="旧回答")
        await session.commit()

    existing = [HumanMessage(content="旧问题", id="db:1"), AIMessage(content="旧回答")]
    final = {"messages": existing + [HumanMessage(content="新问题", id="db:3"), AIMessage(content="新回答")]}
    graph = FakeGraph([], final, existing=existing)
    await _collect(_service(graph, session_factory), str(cid), "新问题")
    sent = graph.inputs[0]["messages"]
    assert [m.content for m in sent] == ["新问题"]


@pytest.mark.asyncio
async def test_second_turn_persists_only_new_messages(session_factory):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        await repo.add_message(cid, role="user", content="旧问题")
        await repo.add_message(cid, role="assistant", content="旧回答")
        await session.commit()

    existing = [HumanMessage(content="旧问题", id="db:1"), AIMessage(content="旧回答")]
    final = {"messages": existing + [HumanMessage(content="新问题", id="db:3"), AIMessage(content="新回答")]}
    graph = FakeGraph([], final, existing=existing)
    await _collect(_service(graph, session_factory), str(cid), "新问题")

    async with session_factory() as session:
        rows = await ChatRepository(session).list_messages(cid)
    assert [r.content for r in rows] == ["旧问题", "旧回答", "新问题", "新回答"]


@pytest.mark.asyncio
async def test_turn_msg_offset_counts_existing_checkpoint_messages(session_factory):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        await session.commit()

    existing = [HumanMessage(content="旧问题", id="db:1"), AIMessage(content="旧回答")]
    final = {"messages": existing + [HumanMessage(content="新问题", id="db:3"), AIMessage(content="新回答")]}
    graph = FakeGraph([], final, existing=existing)
    await _collect(_service(graph, session_factory), str(cid), "新问题")

    # 图内 log_turn 从该偏移之后统计本轮工具调用，必须跳过 checkpoint 里的历史
    assert graph.inputs[0]["turn_msg_offset"] == 2


@pytest.mark.asyncio
async def test_graph_error_emits_error(session_factory):
    class BoomGraph(FakeGraph):
        async def astream(self, input, config, context, stream_mode):
            raise RuntimeError("graph down")
            yield  # pragma: no cover

    events = await _collect(_service(BoomGraph([], {}), session_factory), None, "你好")
    assert events[-1]["type"] == "error"
    assert "graph down" in events[-1]["message"]


@pytest.mark.asyncio
async def test_run_returns_eval_payload(session_factory):
    final = {
        "messages": [HumanMessage(content="你好", id="db:1"), AIMessage(content="您好")],
        "answer": "您好",
        "intent": "chitchat",
        "route": "chitchat",
        "suggested_actions": [],
        "trace": {"forced_rag": False},
    }
    graph = FakeGraph([], final)
    payload = await _service(graph, session_factory).run(None, "你好")
    assert payload["answer"] == "您好"
    assert payload["intent"] == "chitchat"
    assert payload["session_id"] > 0


async def test_chat_router_empty_message_422():
    app = FastAPI()
    app.include_router(chat_module.router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"message": ""})
    assert response.status_code == 422
