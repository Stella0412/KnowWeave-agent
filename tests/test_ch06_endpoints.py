import json

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models import RefundRequest
from app.db.repository import ChatRepository, RefundRepository
from app.db.seed import seed_orders
from app.routers import chat as chat_module


class FakeService:
    async def validate_resume(self, session_id, order_no):
        if order_no == "9999":
            raise ValueError("订单不存在")

    async def resume_stream(self, session_id, order_no):
        yield {"type": "token", "content": f"选中{order_no}"}
        yield {"type": "done"}

    async def run(self, session_id, message):
        return {
            "session_id": 1,
            "interrupted": True,
            "actions": [{"type": "select_order", "orders": [{"order_no": "1001"}]}],
        }

    async def run_resume(self, session_id, order_no):
        if order_no == "9999":
            raise ValueError("订单不存在")
        return {"session_id": int(session_id), "interrupted": False, "answer": order_no}


def _app():
    app = FastAPI()
    app.include_router(chat_module.router)
    return app


async def _post(path, payload):
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, json=payload)


@pytest.mark.asyncio
async def test_agent_returns_interrupt_actions(monkeypatch):
    monkeypatch.setattr(chat_module, "_get_chat_service", lambda request: FakeService())
    resp = await _post("/chat/agent", {"message": "我要退款"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["interrupted"] is True
    assert body["actions"][0]["type"] == "select_order"


@pytest.mark.asyncio
async def test_agent_resume_returns_answer(monkeypatch):
    monkeypatch.setattr(chat_module, "_get_chat_service", lambda request: FakeService())
    resp = await _post(
        "/chat/agent/resume", {"session_id": "1", "order_no": "1001"}
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "1001"


@pytest.mark.asyncio
async def test_agent_resume_rejects_unknown_order(monkeypatch):
    monkeypatch.setattr(chat_module, "_get_chat_service", lambda request: FakeService())
    resp = await _post(
        "/chat/agent/resume", {"session_id": "1", "order_no": "9999"}
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_resume_rejects_unknown_order_before_streaming(monkeypatch):
    monkeypatch.setattr(chat_module, "_get_chat_service", lambda request: FakeService())
    resp = await _post("/chat/resume", {"session_id": "1", "order_no": "9999"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_resume_streams_successfully(monkeypatch):
    monkeypatch.setattr(chat_module, "_get_chat_service", lambda request: FakeService())
    resp = await _post("/chat/resume", {"session_id": "1", "order_no": "1001"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[len("data: "):])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[0] == {"type": "token", "content": "选中1001"}
    assert events[-1] == {"type": "done"}


@pytest.mark.asyncio
async def test_refund_rejects_unknown_reason():
    resp = await _post("/chat/refund", {
        "session_id": "1", "order_no": "1001", "reason_category": "随便写",
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_refund_rejects_unknown_order(monkeypatch, session_factory):
    async with session_factory() as session:
        conversation_id = await ChatRepository(session).create_conversation()
        await session.commit()
    monkeypatch.setattr(chat_module, "SessionFactory", session_factory)
    resp = await _post("/chat/refund", {
        "session_id": str(conversation_id),
        "order_no": "9999",
        "reason_category": "七天无理由",
    })
    assert resp.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("order_no", ["1002", "1003", "1004", "1005"])
async def test_refund_rejects_ineligible_order(
    monkeypatch, session_factory, order_no
):
    # 定制/生鲜/未签收/已取消的订单不能提交退款申请；前端即使被绕过也要挡住。
    await seed_orders(session_factory)
    async with session_factory() as session:
        conversation_id = await ChatRepository(session).create_conversation()
        await session.commit()
    monkeypatch.setattr(chat_module, "SessionFactory", session_factory)
    resp = await _post("/chat/refund", {
        "session_id": str(conversation_id),
        "order_no": order_no,
        "reason_category": "七天无理由",
    })
    assert resp.status_code == 400
    assert "不能提交退款申请" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_refund_returns_existing_pending_without_duplicate(
    monkeypatch, session_factory
):
    await seed_orders(session_factory)
    async with session_factory() as session:
        conversation_id = await ChatRepository(session).create_conversation()
        await session.commit()
    monkeypatch.setattr(chat_module, "SessionFactory", session_factory)
    payload = {
        "session_id": str(conversation_id),
        "order_no": "1001",
        "reason_category": "七天无理由",
    }
    first = await _post("/chat/refund", payload)
    second = await _post("/chat/refund", payload)
    assert first.status_code == 200
    assert first.json()["refund_no"] == second.json()["refund_no"]
    async with session_factory() as session:
        rows = list((await session.execute(select(RefundRequest))).scalars())
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_refund_retries_once_on_refund_no_collision(monkeypatch, session_factory):
    """并发建单时 refund_no 可能撞号；撞号后应重试而不是 500。"""
    await seed_orders(session_factory)
    async with session_factory() as session:
        conversation_id = await ChatRepository(session).create_conversation()
        await session.commit()
    monkeypatch.setattr(chat_module, "SessionFactory", session_factory)

    original = RefundRepository.create
    calls = {"n": 0}

    async def flaky_create(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise IntegrityError("INSERT refund_requests", {}, Exception("duplicate"))
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(RefundRepository, "create", flaky_create)
    resp = await _post("/chat/refund", {
        "session_id": str(conversation_id),
        "order_no": "1001",
        "reason_category": "七天无理由",
    })

    assert resp.status_code == 200
    assert calls["n"] == 2
    async with session_factory() as session:
        rows = list((await session.execute(select(RefundRequest))).scalars())
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_refund_rejects_overlong_note():
    resp = await _post("/chat/refund", {
        "session_id": "1",
        "order_no": "1001",
        "reason_category": "七天无理由",
        "note": "x" * 501,
    })
    assert resp.status_code == 422
