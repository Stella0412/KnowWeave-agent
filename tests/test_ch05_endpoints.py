from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from app.config import Settings
from app.db.models import ConversationStatus, Ticket
from app.db.repository import ChatRepository
from app.graph.build import build_graph
from app.routers import chat as chat_module
from app.services.chat_service import ChatService
from app.tools.audit import write_audit
from app.tools.builtin import register_builtins
from app.tools.executor import ToolEngine
from app.tools.policy import PolicyStore
from app.tools.registry import ToolRegistry, tool_timeout


class FakeService:
    async def stream(self, session_id, message):
        yield {"type": "session", "session_id": "7"}
        yield {"type": "token", "content": "您好"}
        yield {"type": "done"}

    async def run(self, session_id, message):
        return {
            "session_id": 7,
            "answer": "您好",
            "intent": "chitchat",
            "route": "chitchat",
            "tool_calls": [],
            "tool_results": [],
            "suggested_actions": [],
            "trace": {"forced_rag": False},
        }


def _app():
    app = FastAPI()
    app.include_router(chat_module.router)
    return app


class _UnusedModel:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        raise AssertionError("模型不应被 /chat/ticket 按钮调用")


def _ticket_app(session_factory):
    """旧 /chat/ticket 按钮现在走真实 ChatService + 引擎，模型不参与。"""
    registry = ToolRegistry()
    register_builtins(registry)
    engine = ToolEngine(
        PolicyStore(""),
        lambda record: write_audit(session_factory, record),
        timeout=lambda name: tool_timeout(name, 5.0),
        retries=1,
    )
    service = ChatService(
        model=_UnusedModel(), intent_model=_UnusedModel(), session_factory=session_factory,
        graph=build_graph(checkpointer=InMemorySaver()), settings=Settings(_env_file=None),
        registry=registry, engine=engine,
    )
    app = FastAPI()
    app.include_router(chat_module.router)
    app.state.chat_service = service
    return app


@pytest.mark.asyncio
async def test_chat_stream_keeps_existing_contract(monkeypatch):
    monkeypatch.setattr(chat_module, "_get_chat_service", lambda request: FakeService())
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"message": "你好"})
    assert response.status_code == 200
    assert '"type": "token"' in response.text
    assert '"type": "done"' in response.text


@pytest.mark.asyncio
async def test_agent_endpoint_returns_payload(monkeypatch):
    monkeypatch.setattr(chat_module, "_get_chat_service", lambda request: FakeService())
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat/agent", json={"message": "你好"})
    assert response.status_code == 200
    assert response.json()["answer"] == "您好"


@pytest.mark.asyncio
async def test_ticket_endpoint_writes_without_escalating(session_factory):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        await repo.add_message(cid, role="user", content="商品破损")
        await session.commit()

    transport = httpx.ASGITransport(app=_ticket_app(session_factory))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/ticket",
            json={"session_id": str(cid), "ticket_type": "投诉", "description": "商品破损"},
        )
    assert response.status_code == 200
    assert response.json()["ticket_no"].startswith("T8")
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
        tickets = (await session.execute(select(Ticket))).scalars().all()
    assert conv.status is ConversationStatus.ACTIVE
    assert len(tickets) == 1


@pytest.mark.asyncio
async def test_ticket_endpoint_rejects_non_numeric_session(session_factory):
    transport = httpx.ASGITransport(app=_ticket_app(session_factory))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/ticket",
            json={"session_id": "abc", "ticket_type": "售后", "description": "x"},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_ticket_endpoint_404_for_missing_conversation(session_factory):
    transport = httpx.ASGITransport(app=_ticket_app(session_factory))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/ticket",
            json={"session_id": "99999", "ticket_type": "售后", "description": "x"},
        )
    assert response.status_code == 404
