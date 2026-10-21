from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app.graph.agent import agent_llm, agent_tools, build_agent_system_prompt
from app.tools.business import REFUND_REASON_CATEGORIES


class FakeAgent:
    def __init__(self):
        self.bound = []
        self.calls = []

    def bind_tools(self, tools):
        self.bound = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="答")


def _runtime(session_factory):
    return SimpleNamespace(
        context=SimpleNamespace(
            session_factory=session_factory,
            token_budget=4096,
            tool_timeout=5.0,
            tool_retries=1,
            turn=SimpleNamespace(citations=[]),
        )
    )


def _state(**overrides):
    state = {
        "messages": [],
        "conversation_id": 1,
        "intent": "refund_return",
        "route": "refund_flow",
        "order_no": "1001",
        "order_data": {"order_no": "1001", "product_name": "猫爬架"},
        "evidence": "[1] 七天无理由",
        "steps": 0,
        "tokens_used": 0,
        "suggested_actions": [],
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_refund_agent_binds_only_refund_tools(session_factory):
    model = FakeAgent()
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            model=model,
            session_factory=session_factory,
            token_budget=4096,
            tool_timeout=5.0,
            tool_retries=1,
            turn=SimpleNamespace(citations=[]),
        )
    )
    await agent_llm(_state(), runtime)
    assert "submit_refund" in model.bound
    assert "create_ticket" in model.bound
    assert "query_order" not in model.bound
    assert "query_faq" not in model.bound


def test_refund_prompt_contains_order_and_policy():
    text = build_agent_system_prompt(
        "refund_flow", "[1] 七天无理由",
        intent="refund_return", order_data={"order_no": "1001", "product_name": "猫爬架"},
    )
    assert "1001" in text
    assert "猫爬架" in text
    assert "七天无理由" in text
    assert "不要追问退款原因" in text


def test_after_sale_prompt_uses_after_sale_addendum():
    text = build_agent_system_prompt(
        "refund_flow", "[1] 保修条款",
        intent="after_sale", order_data={"order_no": "1001"},
    )
    assert "售后判定" in text
    assert "不要调用 submit_refund" in text


@pytest.mark.asyncio
async def test_after_sale_agent_does_not_bind_submit_refund(session_factory):
    model = FakeAgent()
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            model=model,
            session_factory=session_factory,
            token_budget=4096,
            tool_timeout=5.0,
            tool_retries=1,
            turn=SimpleNamespace(citations=[]),
        )
    )
    await agent_llm(_state(intent="after_sale"), runtime)
    assert "submit_refund" not in model.bound
    assert "create_ticket" in model.bound
    assert "query_order" not in model.bound
    assert "query_faq" not in model.bound


@pytest.mark.asyncio
async def test_after_sale_create_ticket_adds_transfer_action(session_factory):
    state = _state(
        intent="after_sale",
        messages=[AIMessage(content="", tool_calls=[{
            "name": "create_ticket",
            "args": {"description": "猫爬架破损", "ticket_type": "售后"},
            "id": "c1", "type": "tool_call",
        }])],
    )
    update = await agent_tools(state, _runtime(session_factory))
    types = [item["type"] for item in update["suggested_actions"]]
    assert "transfer_human" in types
    assert "create_ticket" in types
    assert "refund_form" not in types


@pytest.mark.asyncio
async def test_submit_refund_intercept_adds_form(session_factory):
    state = _state(
        messages=[AIMessage(content="", tool_calls=[{
            "name": "submit_refund", "args": {"order_no": "1001"},
            "id": "c1", "type": "tool_call",
        }])]
    )
    update = await agent_tools(state, _runtime(session_factory))
    action = update["suggested_actions"][0]
    assert action["type"] == "refund_form"
    assert action["draft"]["order_no"] == "1001"
    assert tuple(action["draft"]["reason_options"]) == REFUND_REASON_CATEGORIES
    assert update["refund_offered"] is True


@pytest.mark.asyncio
async def test_submit_refund_wrong_order_does_not_add_form(session_factory):
    state = _state(
        messages=[AIMessage(content="", tool_calls=[{
            "name": "submit_refund", "args": {"order_no": "9999"},
            "id": "c1", "type": "tool_call",
        }])]
    )
    update = await agent_tools(state, _runtime(session_factory))
    assert update["suggested_actions"] == []
    assert update["refund_offered"] is False
