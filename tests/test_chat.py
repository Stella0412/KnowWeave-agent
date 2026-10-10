import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage

from app.db.models import Conversation
from app.routers import chat as chat_module
from app.services.chat_service import ChatService


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeBound:
    def __init__(self, message):
        self._message = message

    async def ainvoke(self, messages):
        return self._message


class FakeModel:
    def __init__(self, decision=None, chunks=()):
        self._decision = decision or AIMessage(content="直接回答")
        self._chunks = chunks or []

    def bind_tools(self, tools):
        return FakeBound(self._decision)

    def astream(self, messages):
        async def gen():
            for c in self._chunks:
                yield FakeChunk(c)
        return gen()


async def _collect(svc, session_id, message):
    events = []
    async for e in svc.stream(session_id, message):
        events.append(e)
    return events


@pytest.mark.asyncio
async def test_new_session_emits_session_and_persists(session_factory):
    svc = ChatService(FakeModel(), session_factory)
    events = await _collect(svc, None, "你好")
    assert events[0]["type"] == "session"
    assert events[-1]["type"] == "done"
    cid = int(events[0]["session_id"])
    async with session_factory() as session:
        conv = await session.get(Conversation, cid)
    assert conv is not None


@pytest.mark.asyncio
async def test_no_tool_path_streams_and_persists(session_factory):
    svc = ChatService(FakeModel(chunks=["您", "好"]), session_factory)
    events = await _collect(svc, None, "你好")
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == "您好"


@pytest.mark.asyncio
async def test_tool_path_full_roundtrip(session_factory):
    decision = AIMessage(
        content="",
        tool_calls=[
            {"name": "query_logistics", "args": {"order_id": "1001"},
             "id": "call_1", "type": "tool_call"}
        ],
    )
    svc = ChatService(FakeModel(decision=decision, chunks=["您的", "包裹"]), session_factory)
    events = await _collect(svc, None, "订单1001到哪了")
    types = [e["type"] for e in events]
    assert "tool_status" in types
    ts = next(e for e in events if e["type"] == "tool_status")
    assert ts["name"] == "query_logistics"
    # 落库：user / assistant(tool_calls) / tool / assistant(全文) 四类齐全
    cid = int(events[0]["session_id"])
    async with session_factory() as session:
        from app.db.repository import ChatRepository
        rows = await ChatRepository(session).list_messages(cid)
    roles = [r.role.value for r in rows]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert rows[1].tool_calls[0]["name"] == "query_logistics"
    assert rows[2].tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_db_unavailable_emits_error(session_factory):
    class BoomFactory:
        def __call__(self):
            raise RuntimeError("db down")

    svc = ChatService(FakeModel(chunks=["x"]), BoomFactory())
    events = await _collect(svc, "1", "你好")
    assert events[-1]["type"] == "error"


async def test_chat_router_empty_message_422():
    app = FastAPI()
    app.include_router(chat_module.router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/chat", json={"message": ""})
    assert r.status_code == 422
