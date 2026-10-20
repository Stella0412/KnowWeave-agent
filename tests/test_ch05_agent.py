from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

import app.graph.agent as agent_module
from app.db.repository import ChatRepository
from app.graph.agent import agent_llm, agent_tools, build_agent_system_prompt
from app.graph.routing import make_should_continue
from app.rag.pipeline import TurnContext


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
        self.bound_tools = [t.name for t in tools]
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


def test_agent_system_prompt_includes_evidence_and_route_rules():
    knowledge = build_agent_system_prompt("knowledge", "[1] 7 天无理由退货")
    assert "7 天无理由退货" in knowledge
    assert "不要再调用 query_faq" in knowledge
    business = build_agent_system_prompt("business", None)
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
async def test_knowledge_route_excludes_query_faq(monkeypatch):
    captured = {}

    def fake_build_tools(session_factory, conversation_id, turn=None, *, include_faq=True):
        captured["include_faq"] = include_faq
        return []

    monkeypatch.setattr(agent_module, "build_tools", fake_build_tools)
    runtime = _runtime()
    await agent_llm(_state(route="knowledge", evidence="[1] 证据"), runtime)
    assert captured["include_faq"] is False


@pytest.mark.asyncio
async def test_agent_tools_executes_business_tool(session_factory):
    state = _state(messages=[
        HumanMessage(content="订单1001到哪了"),
        AIMessage(content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}
        ]),
    ])
    runtime = _runtime(session_factory)
    update = await agent_tools(state, runtime)
    assert isinstance(update["messages"][0], ToolMessage)
    assert "1001" in update["messages"][0].content


@pytest.mark.asyncio
async def test_agent_tools_intercepts_create_ticket(session_factory):
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
    )
    update = await agent_tools(state, _runtime(session_factory))
    assert update["suggested_actions"][0]["type"] == "create_ticket"
    assert update["suggested_actions"][0]["draft"]["description"] == "商品破损"
    assert "已呈现给用户" in update["messages"][0].content
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
    ])
    update = await agent_tools(state, _runtime(session_factory))
    assert update["suggested_actions"] == []
    assert "参数校验失败" in update["messages"][0].content


@pytest.mark.asyncio
async def test_agent_tools_dedupes_ticket_actions(session_factory):
    call = {
        "name": "create_ticket",
        "args": {"description": "商品破损", "ticket_type": "售后"},
        "id": "c1", "type": "tool_call",
    }
    state = _state(
        route="business",
        messages=[HumanMessage(content="建工单"), AIMessage(content="", tool_calls=[call, {**call, "id": "c2"}])],
    )
    update = await agent_tools(state, _runtime(session_factory))
    assert len(update["suggested_actions"]) == 1


@pytest.mark.asyncio
async def test_agent_tools_keeps_existing_citations(session_factory, monkeypatch):
    @tool
    async def query_probe(keyword: str) -> str:
        """探测工具。"""
        return "ok"

    runtime = _runtime(session_factory)
    runtime.context.turn.citations.append({"n": 1, "chunk_id": 7})

    def fake_build_tools(*args, **kwargs):
        return [query_probe]

    monkeypatch.setattr(agent_module, "build_tools", fake_build_tools)
    state = _state(
        route="business",
        messages=[HumanMessage(content="查"), AIMessage(content="", tool_calls=[
            {"name": "query_probe", "args": {"keyword": "x"}, "id": "c1", "type": "tool_call"}
        ])],
        citations=[{"n": 2, "chunk_id": 9}],
    )
    update = await agent_tools(state, runtime)
    assert {c["n"] for c in update["citations"]} == {1, 2}
