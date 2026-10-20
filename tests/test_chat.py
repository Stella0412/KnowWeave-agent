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
    assert "citations" not in [e["type"] for e in events]


@pytest.mark.asyncio
async def test_tool_path_full_roundtrip(session_factory):
    decision = AIMessage(
        content="",
        tool_calls=[
            {"name": "query_logistics", "args": {"tracking_no": "SF1001CN"},
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


@pytest.mark.asyncio
async def test_tool_path_pushes_citations_before_done(session_factory, monkeypatch):
    """验收③ 依赖这条 SSE 契约：引用载荷必须在 done 之前到达，前端才来得及挂角标。

    这条同时钉住两件事：build_tools 真的收到了第三个实参（TurnContext），
    以及 ChatService 把 turn.citations 推成了 citations 事件。
    """
    import app.services.chat_service as chat_service
    from langchain_core.tools import tool

    captured = {}

    @tool
    async def query_probe(keyword: str) -> str:
        """探测工具：把一条引用写进本轮的侧信道。"""
        captured["turn"].citations.append({"n": 1, "chunk_id": 7, "answer": "满 99 元包邮"})
        return "命中"

    def fake_build_tools(session_factory, conversation_id, turn=None):
        captured["turn"] = turn
        return [query_probe]

    monkeypatch.setattr(chat_service, "build_tools", fake_build_tools)

    decision = AIMessage(content="", tool_calls=[
        {"name": "query_probe", "args": {"keyword": "邮费"}, "id": "call_1", "type": "tool_call"}
    ])
    svc = ChatService(FakeModel(decision=decision, chunks=["好的"]), session_factory)
    events = await _collect(svc, None, "邮费多少")
    types = [e["type"] for e in events]

    assert "citations" in types
    assert types.index("citations") < types.index("done")   # 顺序，不是只要出现过
    assert types[-1] == "done"
    payload = next(e for e in events if e["type"] == "citations")
    assert payload["items"][0]["chunk_id"] == 7
    # 侧信道里必须是用户原话（DDL 的语义 + T13 的落池规则都建立在这上面）
    assert captured["turn"].raw_question == "邮费多少"
