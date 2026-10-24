"""ch08 · Task 9 工单恢复 HTTP 接口与历史 pending_action 集成测试。

用真实 ChatService + 真实图 + 脚本模型 + 真实 session_factory 走通整个 HTTP 链路；
tickets 与 ToolAuditLog 一律通过真实 session_factory 查询，不 mock 成固定答案。
"""

import asyncio
import re
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from app.config import Settings
from app.db.models import Ticket, ToolAuditLog
from app.graph.build import build_graph
from app.prompts.orchestrator import IntentOutput
from app.routers import chat as chat_module
from app.routers import conversations as conversations_module
from app.services.chat_service import ChatService
from app.tools.audit import write_audit
from app.tools.builtin import register_builtins
from app.tools.executor import ToolEngine
from app.tools.policy import PolicyStore
from app.tools.registry import ToolRegistry, tool_timeout

TICKET_NO_RE = re.compile(r"T8[0-9a-f]{30}")


class TicketScriptedModel:
    """第一次 agent_llm 调 create_ticket；之后读 ToolMessage 里的工单号回给用户。"""

    def __init__(self):
        self.calls = 0
        self.bound = None

    def bind_tools(self, tools):
        self.bound = [t["name"] for t in tools]
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "create_ticket",
                "args": {"description": "收到的猫粮包装破损", "ticket_type": "售后"},
                "id": "c1", "type": "tool_call",
            }])
        ticket_no = ""
        for message in messages:
            if isinstance(message, ToolMessage):
                content = message.content if isinstance(message.content, str) else str(message.content)
                match = TICKET_NO_RE.search(content)
                if match:
                    ticket_no = match.group(0)
        return AIMessage(content=f"已为您创建工单，工单号 {ticket_no}")


class FakeIntentModel:
    def __init__(self, intent):
        self.intent = intent

    async def ainvoke(self, messages):
        return IntentOutput(intent=self.intent, confidence=0.9)


async def _ticket_count(session_factory) -> int:
    async with session_factory() as session:
        return len(list((await session.execute(select(Ticket))).scalars()))


async def _audit_rows(session_factory) -> list[ToolAuditLog]:
    async with session_factory() as session:
        return list((await session.execute(select(ToolAuditLog).order_by(ToolAuditLog.id))).scalars())


@pytest.fixture
def stack(session_factory, monkeypatch):
    registry = ToolRegistry()
    register_builtins(registry)
    policy = PolicyStore("")
    engine = ToolEngine(
        policy,
        lambda record: write_audit(session_factory, record),
        timeout=lambda name: tool_timeout(name, 5.0),
        retries=1,
    )
    graph = build_graph(checkpointer=InMemorySaver())
    model = TicketScriptedModel()
    service = ChatService(
        model=model, intent_model=FakeIntentModel("order"), session_factory=session_factory,
        graph=graph, settings=Settings(_env_file=None), registry=registry, engine=engine,
    )
    app = FastAPI()
    app.include_router(chat_module.router)
    app.include_router(conversations_module.router)
    app.state.chat_service = service
    monkeypatch.setattr(conversations_module, "SessionFactory", session_factory)
    return SimpleNamespace(app=app, service=service, graph=graph, session_factory=session_factory, model=model)


def _client(stack):
    transport = httpx.ASGITransport(app=stack.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def _open_ticket_interrupt(stack):
    """发一句建单请求，返回 (conversation_id, confirmation_id)。"""
    async with _client(stack) as client:
        resp = await client.post("/chat/agent", json={"message": "帮我建个工单"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["interrupted"] is True
    action = body["actions"][0]
    assert action["type"] == "ticket_preview"
    return body["session_id"], action["confirmation_id"]


async def _post_resume(stack, payload):
    async with _client(stack) as client:
        return await client.post("/chat/ticket/resume", json=payload)


@pytest.mark.asyncio
async def test_no_pending_confirmation_rejected_without_tickets_or_audit(stack):
    async with stack.session_factory() as session:
        from app.db.repository import ChatRepository
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    resp = await _post_resume(stack, {
        "session_id": str(cid), "confirmation_id": "TCnope", "decision": "confirm",
    })
    assert resp.status_code == 409
    assert await _ticket_count(stack.session_factory) == 0
    assert await _audit_rows(stack.session_factory) == []


@pytest.mark.asyncio
async def test_wrong_session_and_unknown_session_rejected(stack):
    cid, confirmation_id = await _open_ticket_interrupt(stack)
    async with stack.session_factory() as session:
        from app.db.repository import ChatRepository
        other = await ChatRepository(session).create_conversation()
        await session.commit()
    wrong = await _post_resume(stack, {
        "session_id": str(other), "confirmation_id": confirmation_id, "decision": "confirm",
    })
    assert wrong.status_code == 409
    missing = await _post_resume(stack, {
        "session_id": "999999", "confirmation_id": confirmation_id, "decision": "confirm",
    })
    assert missing.status_code == 404
    assert await _ticket_count(stack.session_factory) == 0


@pytest.mark.asyncio
async def test_old_confirmation_id_rejected(stack):
    cid, confirmation_id = await _open_ticket_interrupt(stack)
    resp = await _post_resume(stack, {
        "session_id": str(cid), "confirmation_id": "TC" + "0" * 30, "decision": "confirm",
    })
    assert resp.status_code == 409
    assert "失效" in resp.json()["detail"]
    assert await _ticket_count(stack.session_factory) == 0
    assert await _audit_rows(stack.session_factory) == []


@pytest.mark.asyncio
async def test_client_supplied_description_is_422(stack):
    cid, confirmation_id = await _open_ticket_interrupt(stack)
    resp = await _post_resume(stack, {
        "session_id": str(cid),
        "confirmation_id": confirmation_id,
        "decision": "confirm",
        "description": "篡改描述",
    })
    assert resp.status_code == 422
    assert await _ticket_count(stack.session_factory) == 0


@pytest.mark.asyncio
async def test_non_numeric_session_is_400(stack):
    resp = await _post_resume(stack, {
        "session_id": "abc", "confirmation_id": "TCx", "decision": "confirm",
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_cancel_terminates_with_permission_denied_audit_and_zero_tickets(stack):
    cid, confirmation_id = await _open_ticket_interrupt(stack)
    before = await _ticket_count(stack.session_factory)
    resp = await _post_resume(stack, {
        "session_id": str(cid), "confirmation_id": confirmation_id, "decision": "cancel",
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    audits = await _audit_rows(stack.session_factory)
    assert len(audits) == 1
    assert audits[0].status.value == "权限拒绝"
    assert audits[0].error_message == "用户取消确认"
    assert await _ticket_count(stack.session_factory) == before == 0


@pytest.mark.asyncio
async def test_confirm_streams_real_ticket_number_and_writes_one_ticket(stack):
    cid, confirmation_id = await _open_ticket_interrupt(stack)
    resp = await _post_resume(stack, {
        "session_id": str(cid), "confirmation_id": confirmation_id, "decision": "confirm",
    })
    assert resp.status_code == 200
    async with stack.session_factory() as session:
        tickets = list((await session.execute(select(Ticket))).scalars())
    assert len(tickets) == 1
    assert tickets[0].ticket_no in resp.text
    audits = await _audit_rows(stack.session_factory)
    assert len(audits) == 1
    assert audits[0].status.value == "成功"


@pytest.mark.asyncio
async def test_history_pending_action_persists_then_clears(stack):
    cid, confirmation_id = await _open_ticket_interrupt(stack)
    async with _client(stack) as client:
        before = await client.get(f"/api/conversations/{cid}/messages")
    assert before.status_code == 200
    pending = before.json()["pending_action"]
    assert pending is not None
    assert pending["type"] == "ticket_preview"
    assert pending["confirmation_id"] == confirmation_id
    assert pending["draft"]["ticket_type"] == "售后"

    async with _client(stack) as client:
        await client.post("/chat/ticket/resume", json={
            "session_id": str(cid), "confirmation_id": confirmation_id, "decision": "cancel",
        })
        after = await client.get(f"/api/conversations/{cid}/messages")
    assert after.json()["pending_action"] is None


@pytest.mark.asyncio
async def test_concurrent_same_conversation_resume_is_serialized(stack):
    cid, confirmation_id = await _open_ticket_interrupt(stack)
    payload = {"session_id": str(cid), "confirmation_id": confirmation_id, "decision": "confirm"}
    async with _client(stack) as client:
        first, second = await asyncio.gather(
            client.post("/chat/ticket/resume", json=payload),
            client.post("/chat/ticket/resume", json=payload),
        )
    assert sorted([first.status_code, second.status_code]) == [200, 409]
    assert await _ticket_count(stack.session_factory) == 1


@pytest.mark.asyncio
async def test_failed_resume_releases_lock_and_next_resume_proceeds(stack):
    cid, confirmation_id = await _open_ticket_interrupt(stack)
    stale = await _post_resume(stack, {
        "session_id": str(cid), "confirmation_id": "TC" + "0" * 30, "decision": "confirm",
    })
    assert stale.status_code == 409
    # 校验失败必须释放会话锁，否则后续正确恢复会被永久阻塞。
    ok = await _post_resume(stack, {
        "session_id": str(cid), "confirmation_id": confirmation_id, "decision": "confirm",
    })
    assert ok.status_code == 200
    assert await _ticket_count(stack.session_factory) == 1
