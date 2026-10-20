from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from app.db.models import ConversationStatus
from app.db.repository import ChatRepository
from app.routers import chat as chat_module


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
async def test_ticket_endpoint_writes_without_escalating(monkeypatch, session_factory):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        await session.commit()

    monkeypatch.setattr(chat_module, "SessionFactory", session_factory)
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/ticket",
            json={"session_id": str(cid), "ticket_type": "投诉", "description": "商品破损"},
        )
    assert response.status_code == 200
    assert response.json()["ticket_no"].startswith("T")
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
    assert conv.status is ConversationStatus.ACTIVE


@pytest.mark.asyncio
async def test_ticket_endpoint_rejects_non_numeric_session():
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/ticket",
            json={"session_id": "abc", "ticket_type": "售后", "description": "x"},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_ticket_endpoint_404_for_missing_conversation(monkeypatch, session_factory):
    monkeypatch.setattr(chat_module, "SessionFactory", session_factory)
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/ticket",
            json={"session_id": "99999", "ticket_type": "售后", "description": "x"},
        )
    assert response.status_code == 404
