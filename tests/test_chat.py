import httpx
import pytest
from fastapi import FastAPI

from app.routers import chat as chat_module
from app.services.chat_service import ChatService
from app.services.context_service import ContextStore


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, chunks):
        self._chunks = chunks

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
async def test_stream_generates_session_when_none():
    svc = ChatService(FakeModel(["您", "好"]), ContextStore())
    events = await _collect(svc, None, "你好")
    assert events[0]["type"] == "session"
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_stream_two_turns_share_context():
    store = ContextStore()
    svc = ChatService(FakeModel(["a1"]), store)
    await _collect(svc, "s1", "第一问")
    await _collect(svc, "s1", "第二问")
    msgs = store.get_messages("s1")
    # 两轮 = 2 条用户 + 2 条助手
    assert len(msgs) == 4


@pytest.mark.asyncio
async def test_stream_emits_error_event_on_model_failure():
    class Boom:
        def astream(self, messages):
            async def gen():
                raise RuntimeError("upstream down")
                yield  # pragma: no cover
            return gen()

    svc = ChatService(Boom(), ContextStore())
    events = await _collect(svc, "s1", "你好")
    assert events[-1]["type"] == "error"


async def test_chat_router_empty_message_422():
    app = FastAPI()
    app.include_router(chat_module.router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/chat", json={"message": ""})
    assert r.status_code == 422
