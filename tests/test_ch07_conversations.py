import httpx
import pytest
from fastapi import FastAPI

from app.db.models import MessageRole
from app.db.repository import ChatRepository
from app.routers import conversations as conversations_module


def _app():
    app = FastAPI()
    app.include_router(conversations_module.router)
    return app


async def _get(path):
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_lists_conversations_newest_first_with_preview_and_flags(
    monkeypatch, session_factory,
):
    monkeypatch.setattr(conversations_module, "SessionFactory", session_factory)
    async with session_factory() as session:
        repo = ChatRepository(session)
        first = await repo.create_conversation()
        await repo.add_message(first, role="user", content="订单1001到哪了" + "啊" * 50)
        await repo.add_message(first, role="assistant", content="您的包裹已发出")
        await repo.add_message(first, role="tool", content="工具结果", tool_call_id="c1")
        await repo.add_message(first, role="assistant", content="   ")
        second = await repo.create_conversation()
        await repo.add_message(second, role="user", content="我要退款")
        conv = await repo.get_conversation(second)
        conv.summary_upto_msg_id = 99
        await session.commit()
    resp = await _get("/api/conversations")
    assert resp.status_code == 200
    items = resp.json()["conversations"]
    assert [item["id"] for item in items] == [second, first]
    assert items[0]["preview"] == "我要退款"
    assert items[0]["summarized"] is True
    assert items[1]["summarized"] is False
    assert items[1]["message_count"] == 2
    assert len(items[1]["preview"]) == 40


@pytest.mark.asyncio
async def test_messages_filter_tool_and_empty_rows(monkeypatch, session_factory):
    monkeypatch.setattr(conversations_module, "SessionFactory", session_factory)
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        await repo.add_message(cid, role="user", content="问")
        await repo.add_message(cid, role="assistant", content="答")
        await repo.add_message(cid, role="tool", content="工具", tool_call_id="c1")
        await repo.add_message(cid, role="assistant", content="")
        await session.commit()
    resp = await _get(f"/api/conversations/{cid}/messages")
    assert resp.status_code == 200
    assert [m["role"] for m in resp.json()["messages"]] == ["user", "assistant"]
    assert resp.json()["messages"][1]["content"] == "答"


@pytest.mark.asyncio
async def test_messages_returns_404_for_unknown_or_foreign_user(
    monkeypatch, session_factory,
):
    monkeypatch.setattr(conversations_module, "SessionFactory", session_factory)
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation(user_id="someone")
        await session.commit()
    assert (await _get("/api/conversations/9999/messages")).status_code == 404
    assert (await _get(f"/api/conversations/{cid}/messages")).status_code == 404
