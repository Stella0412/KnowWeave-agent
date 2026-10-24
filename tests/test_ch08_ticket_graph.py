"""ch08 · Task 9 工单确认的图状态集成测试（服务级，走真实图 + 真实 DB）。

覆盖：interrupt → Command(resume) 恢复回合、证据消费/取消、写一次、跨类型拒绝、
待确认时拒收新聊天轮次。tickets 与 ToolAuditLog 通过真实 session_factory 查询。
"""

import re
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from app.config import Settings
from app.db.models import Ticket, ToolAuditLog
from app.db.repository import ChatRepository
from app.db.seed import seed_orders
from app.graph.build import build_graph
from app.graph.runtime import thread_config
from app.prompts.orchestrator import IntentOutput
from app.services.chat_service import ChatService, TicketResumeRejected
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


def _build_service(session_factory, intent="order"):
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
        model=model, intent_model=FakeIntentModel(intent), session_factory=session_factory,
        graph=graph, settings=Settings(_env_file=None), registry=registry, engine=engine,
    )
    return SimpleNamespace(service=service, graph=graph, model=model)


async def _ticket_rows(session_factory):
    async with session_factory() as session:
        return list((await session.execute(select(Ticket))).scalars())


async def _audit_rows(session_factory):
    async with session_factory() as session:
        return list((await session.execute(select(ToolAuditLog).order_by(ToolAuditLog.id))).scalars())


async def _graph_ticket_request(graph, conversation_id):
    snapshot = await graph.aget_state(thread_config(conversation_id))
    return (snapshot.values or {}).get("ticket_request")


async def _open_interrupt(service):
    result = await service.run(None, "帮我建个工单")
    assert result["interrupted"] is True
    action = result["actions"][0]
    assert action["type"] == "ticket_preview"
    assert action["confirmation_id"].startswith("TC")
    assert action["draft"]["ticket_type"] == "售后"
    return result["session_id"], action["confirmation_id"]


@pytest.mark.asyncio
async def test_first_run_interrupts_with_ticket_preview_and_writes_nothing(session_factory):
    stack = _build_service(session_factory)
    cid, confirmation_id = await _open_interrupt(stack.service)
    assert stack.model.calls == 1
    assert await _ticket_rows(session_factory) == []
    assert await _audit_rows(session_factory) == []
    evidence = await _graph_ticket_request(stack.graph, cid)
    assert evidence["status"] == "active"


@pytest.mark.asyncio
async def test_confirm_writes_once_consumes_evidence_and_stale_id_rejected(session_factory):
    stack = _build_service(session_factory)
    cid, confirmation_id = await _open_interrupt(stack.service)
    events = [e async for e in stack.service.ticket_resume_stream(str(cid), confirmation_id, "confirm")]
    text = "".join(e.get("content", "") for e in events if e.get("type") == "token")
    tickets = await _ticket_rows(session_factory)
    assert len(tickets) == 1
    assert tickets[0].ticket_no in text
    audits = await _audit_rows(session_factory)
    assert len(audits) == 1
    assert audits[0].status.value == "成功"
    evidence = await _graph_ticket_request(stack.graph, cid)
    assert evidence["status"] == "consumed"

    # 旧 confirmation_id 不能复用：校验期直接拒绝，不重跑节点、不产生第二张工单。
    with pytest.raises(TicketResumeRejected) as exc:
        await stack.service.validate_ticket_resume(str(cid), confirmation_id, "confirm")
    assert exc.value.status_code == 409
    assert len(await _ticket_rows(session_factory)) == 1


@pytest.mark.asyncio
async def test_cancel_writes_nothing_and_cancels_evidence(session_factory):
    stack = _build_service(session_factory)
    cid, confirmation_id = await _open_interrupt(stack.service)
    events = [e async for e in stack.service.ticket_resume_stream(str(cid), confirmation_id, "cancel")]
    assert await _ticket_rows(session_factory) == []
    audits = await _audit_rows(session_factory)
    assert len(audits) == 1
    assert audits[0].status.value == "权限拒绝"
    assert audits[0].error_message == "用户取消确认"
    evidence = await _graph_ticket_request(stack.graph, cid)
    assert evidence["status"] == "cancelled"


@pytest.mark.asyncio
async def test_new_chat_while_pending_is_rejected_before_user_message_persisted(session_factory):
    stack = _build_service(session_factory)
    cid, confirmation_id = await _open_interrupt(stack.service)
    async with session_factory() as session:
        before = len(await ChatRepository(session).list_messages(cid))
    with pytest.raises(ValueError, match="确认或取消"):
        await stack.service.run(str(cid), "继续对话")
    async with session_factory() as session:
        after = len(await ChatRepository(session).list_messages(cid))
    assert after == before


@pytest.mark.asyncio
async def test_abandoned_resume_stream_releases_conversation_lock(session_factory):
    """校验成功后若 SSE 流从未被迭代（客户端在响应开始前断开），会话锁必须兜底释放。"""
    stack = _build_service(session_factory)
    cid, confirmation_id = await _open_interrupt(stack.service)
    guard = await stack.service.validate_ticket_resume(str(cid), confirmation_id, "confirm")
    assert cid in stack.service._held_locks
    # 等价于 SSE 响应 finally 里的兜底释放（流从未接管锁）。
    guard.release_if_unclaimed()
    assert cid not in stack.service._held_locks
    # 后续恢复仍能取到锁并正常建单，而不是被上一次的锁永久阻塞。
    [e async for e in stack.service.ticket_resume_stream(str(cid), confirmation_id, "confirm")]
    assert len(await _ticket_rows(session_factory)) == 1


@pytest.mark.asyncio
async def test_wrong_kind_cross_calls_are_rejected(session_factory):
    await seed_orders(session_factory)
    # 方向一：工单中断不能走订单恢复。
    ticket_stack = _build_service(session_factory, intent="order")
    cid, confirmation_id = await _open_interrupt(ticket_stack.service)
    with pytest.raises(ValueError, match="没有待选择的订单"):
        await ticket_stack.service.validate_resume(str(cid), "1001")

    # 方向二：订单中断不能走工单恢复。
    refund_stack = _build_service(session_factory, intent="refund_return")
    result = await refund_stack.service.run(None, "我要退款")
    assert result["interrupted"] is True
    assert result["actions"][0]["type"] == "select_order"
    with pytest.raises(TicketResumeRejected) as exc:
        await refund_stack.service.validate_ticket_resume(
            str(result["session_id"]), "TCnope", "confirm"
        )
    assert exc.value.status_code == 409
    assert await _ticket_rows(session_factory) == []
