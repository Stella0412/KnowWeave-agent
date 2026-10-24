from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.db.repository import ChatRepository
from app.graph.agent import agent_llm, agent_tools
from app.graph.routing import make_should_continue
from app.prompts.context import build_dynamic_block
from app.rag.pipeline import TurnContext
from app.services.token_counter import TokenCounter


class FakeBound:
    def __init__(self, message):
        self._message = message

    async def ainvoke(self, messages):
        return self._message


class FakeAgentModel:
    def __init__(self, message):
        self._message = message
        self.bound_tools = None
        self.seen = []

    def bind_tools(self, tools):
        self.bound_tools = [t["name"] for t in tools]
        return self

    async def ainvoke(self, messages):
        self.seen.append(messages)
        return self._message


def _runtime(session_factory=None, raw_question="订单1001到哪了"):
    return SimpleNamespace(
        context=SimpleNamespace(
            model=FakeAgentModel(AIMessage(content="答")),
            session_factory=session_factory,
            turn=TurnContext(raw_question=raw_question, conversation_id=1),
            token_budget=4096,
            tool_timeout=5.0,
            tool_retries=1,
            counter=TokenCounter(estimate_factor=1.0),
            tool_result_max_tokens=1200,
            context_manager=None,
            budget=None,
            settings=None,
        )
    )


def _state(**overrides):
    state = {
        "messages": [HumanMessage(content="订单1001到哪了")],
        "conversation_id": 1,
        "route": "business",
        "steps": 0,
        "tokens_used": 0,
        "suggested_actions": [],
        "citations": [],
        "pending_tool_calls": [],
        "tool_cursor": 0,
        "tool_tokens_used": 0,
        "refund_offered": None,
        "trace": {},
    }
    state.update(overrides)
    return state


def test_should_continue_prefers_convergence_then_max_steps():
    should = make_should_continue(2)
    call = {"name": "x", "args": {}, "id": "c1", "type": "tool_call"}
    assert should({"steps": 2, "messages": [AIMessage(content="答")]}) == "log"
    assert should({"steps": 2, "messages": [AIMessage(content="", tool_calls=[call])]}) == "fallback"
    assert should({"steps": 1, "messages": [AIMessage(content="", tool_calls=[call])]}) == "tools"


def test_dynamic_block_includes_evidence_and_route_rules():
    knowledge = build_dynamic_block(
        route="knowledge", intent=None, summary="", evidence="[1] 7 天无理由退货",
        order_data=None, counter=TokenCounter(estimate_factor=1.0),
        evidence_budget_tokens=1000, order_max_tokens=600,
    )
    assert "7 天无理由退货" in knowledge
    assert "不要再调用 query_faq" in knowledge
    business = build_dynamic_block(
        route="business", intent=None, summary="", evidence=None,
        order_data=None, counter=TokenCounter(estimate_factor=1.0),
        evidence_budget_tokens=1000, order_max_tokens=600,
    )
    assert "先调用 query_order 拿到物流单号" in business


@pytest.mark.asyncio
async def test_agent_llm_counts_steps_tokens_and_sets_answer():
    model = FakeAgentModel(AIMessage(content="直接回答"))
    model._message.usage_metadata = {"total_tokens": 12}
    runtime = _runtime()
    runtime.context.model = model
    update = await agent_llm(_state(route="chitchat"), runtime)
    assert update["answer"] == "直接回答"
    assert update["steps"] == 1
    assert update["tokens_used"] == 12
    assert update["stop_reason"] == "no_tool_calls"


@pytest.mark.asyncio
async def test_agent_llm_keeps_answer_none_when_tool_calls():
    message = AIMessage(content="", tool_calls=[
        {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}
    ])
    model = FakeAgentModel(message)
    runtime = _runtime()
    runtime.context.model = model
    update = await agent_llm(_state(), runtime)
    assert update["answer"] is None
    assert update["steps"] == 1
    assert "stop_reason" not in update


@pytest.mark.asyncio
async def test_agent_llm_binds_full_tool_set_on_knowledge_route():
    # 绑定不再按路由裁剪：固定 5 个内置工具逐字稳定（query_logistics 已移交 MCP），
    # 路由差异只由执行端守卫决定。
    model = FakeAgentModel(AIMessage(content="直接回答"))
    runtime = _runtime()
    runtime.context.model = model
    await agent_llm(_state(route="knowledge", evidence="[1] 证据"), runtime)
    assert model.bound_tools == [
        "query_order", "query_product",
        "query_faq", "submit_refund", "create_ticket",
    ]


@pytest.mark.asyncio
async def test_agent_tools_executes_business_tool(session_factory):
    state = _state(messages=[
        HumanMessage(content="订单1001到哪了"),
        AIMessage(content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}
        ]),
    ], pending_tool_calls=[
        {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}
    ], tool_cursor=0)
    runtime = _runtime(session_factory)
    update = await agent_tools(state, runtime)
    assert isinstance(update["messages"][0], ToolMessage)
    assert "1001" in update["messages"][0].content


@pytest.mark.asyncio
async def test_agent_tools_rejects_create_ticket_without_evidence(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    state = _state(
        conversation_id=cid,
        route="business",
        messages=[
            HumanMessage(content="帮我建个工单"),
            AIMessage(content="", tool_calls=[{
                "name": "create_ticket",
                "args": {"description": "商品破损", "ticket_type": "售后"},
                "id": "c1", "type": "tool_call",
            }]),
        ],
        pending_tool_calls=[{
            "name": "create_ticket",
            "args": {"description": "商品破损", "ticket_type": "售后"},
            "id": "c1", "type": "tool_call",
        }],
        tool_cursor=0,
    )
    update = await agent_tools(state, _runtime(session_factory))
    # 新语义：create_ticket 走统一引擎；没有服务端建单证据 → 守卫拦截，不产出任何前端动作。
    assert update["suggested_actions"] == []
    assert "没有有效的用户建单请求证据" in update["messages"][0].content
    from sqlalchemy import select
    from app.db.models import Ticket
    async with session_factory() as session:
        tickets = (await session.execute(select(Ticket))).scalars().all()
    assert tickets == []


@pytest.mark.asyncio
async def test_agent_tools_rejects_bad_ticket_args(session_factory):
    state = _state(messages=[
        HumanMessage(content="建工单"),
        AIMessage(content="", tool_calls=[{
            "name": "create_ticket",
            "args": {"description": "缺类型"},
            "id": "c1", "type": "tool_call",
        }]),
    ], pending_tool_calls=[{
        "name": "create_ticket",
        "args": {"description": "缺类型"},
        "id": "c1", "type": "tool_call",
    }], tool_cursor=0)
    update = await agent_tools(state, _runtime(session_factory))
    assert update["suggested_actions"] == []
    assert "参数校验失败" in update["messages"][0].content


@pytest.mark.asyncio
async def test_agent_tools_replay_same_call_writes_single_ticket(session_factory, monkeypatch):
    # LangGraph 恢复会重跑 agent_tools 节点开头：同一 tool_call_id 重放必须复用稳定工单号，
    # 不能产生第二张工单。这里用可信确认恒通过来直连验证写一次语义。
    async def _confirm(definition, arguments, context):
        context.ticket_decision = "confirm"
        return True

    monkeypatch.setattr("app.graph.agent.confirm_ticket_call", _confirm)
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    call = {
        "name": "create_ticket",
        "args": {"description": "商品破损", "ticket_type": "售后"},
        "id": "c1", "type": "tool_call",
    }
    evidence = {"message_id": 1, "quote": "帮我建个工单", "status": "active"}

    def make_state():
        return _state(
            conversation_id=cid,
            route="business",
            ticket_request=dict(evidence),
            messages=[HumanMessage(content="帮我建个工单"), AIMessage(content="", tool_calls=[call])],
            pending_tool_calls=[call],
            tool_cursor=0,
        )

    first = await agent_tools(make_state(), _runtime(session_factory))
    assert "工单号" in first["messages"][0].content
    assert first["ticket_request"]["status"] == "consumed"

    second = await agent_tools(make_state(), _runtime(session_factory))
    assert second["ticket_request"]["status"] == "consumed"
    from sqlalchemy import select
    from app.db.models import Ticket
    async with session_factory() as session:
        tickets = (await session.execute(select(Ticket))).scalars().all()
    assert len(tickets) == 1


@pytest.mark.asyncio
async def test_agent_tools_keeps_existing_citations(session_factory):
    runtime = _runtime(session_factory)
    runtime.context.turn.citations.append({"n": 1, "chunk_id": 7})
    state = _state(
        route="business",
        messages=[HumanMessage(content="查"), AIMessage(content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1"}, "id": "c1", "type": "tool_call"}
        ])],
        pending_tool_calls=[
            {"name": "query_order", "args": {"order_id": "1"}, "id": "c1", "type": "tool_call"}
        ],
        tool_cursor=0,
        citations=[{"n": 2, "chunk_id": 9}],
    )
    update = await agent_tools(state, runtime)
    assert {c["n"] for c in update["citations"]} == {1, 2}
