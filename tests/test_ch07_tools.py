from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

import app.graph.agent as agent_module
from app.services.token_counter import TokenCounter
from app.tools.registry import allowed_tool_names, build_tools


def test_tool_order_is_static():
    tools = build_tools(None, 1)
    assert [t.name for t in tools] == [
        "query_order", "query_product", "query_logistics",
        "query_faq", "submit_refund", "create_ticket",
    ]


def test_refund_whitelist_only_allows_refund_and_ticket():
    assert allowed_tool_names("refund_flow", "refund_return") == {
        "submit_refund", "create_ticket",
    }
    assert allowed_tool_names("refund_flow", "after_sale") == {"create_ticket"}
    assert "query_faq" not in allowed_tool_names("knowledge", None)


def _runtime(tool_result_max_tokens=1200):
    return SimpleNamespace(
        context=SimpleNamespace(
            session_factory=None,
            token_budget=4096,
            tool_timeout=5.0,
            tool_retries=1,
            turn=SimpleNamespace(citations=[]),
            counter=TokenCounter(estimate_factor=1.0),
            tool_result_max_tokens=tool_result_max_tokens,
        )
    )


def _state(route, intent, calls):
    return {
        "messages": [HumanMessage(content="问"), AIMessage(content="", tool_calls=calls)],
        "conversation_id": 1,
        "route": route,
        "intent": intent,
        "suggested_actions": [],
        "refund_offered": None,
    }


@pytest.mark.asyncio
async def test_blocked_submit_refund_never_creates_form(monkeypatch):
    seen = []

    @tool
    async def query_order(order_id: str) -> str:
        """查订单。"""
        seen.append(order_id)
        return "订单1001"

    @tool
    async def submit_refund(order_no: str) -> str:
        """退款。"""
        return "不应执行"

    @tool
    async def create_ticket(description: str, ticket_type: str) -> str:
        """建单。"""
        return "不应执行"

    monkeypatch.setattr(
        agent_module, "build_tools", lambda *args, **kwargs: [
            query_order, submit_refund, create_ticket,
        ]
    )
    state = _state("business", "refund_return", [{
        "name": "submit_refund", "args": {"order_no": "1001"},
        "id": "c1", "type": "tool_call",
    }])
    update = await agent_module.agent_tools(state, _runtime())
    assert update["suggested_actions"] == []
    assert update["refund_offered"] is False
    assert "本轮不允许调用 submit_refund" in update["messages"][0].content


@pytest.mark.asyncio
async def test_blocked_create_ticket_never_adds_actions(monkeypatch):
    @tool
    async def create_ticket(description: str, ticket_type: str) -> str:
        """建单。"""
        return "不应执行"

    monkeypatch.setattr(
        agent_module, "build_tools", lambda *args, **kwargs: [create_ticket]
    )
    state = _state("knowledge", None, [{
        "name": "submit_refund", "args": {"order_no": "1001"},
        "id": "c1", "type": "tool_call",
    }])
    update = await agent_module.agent_tools(state, runtime := _runtime())
    assert update["suggested_actions"] == []


@pytest.mark.asyncio
async def test_tool_result_total_is_capped(monkeypatch):
    @tool
    async def query_order(order_id: str) -> str:
        """查订单。"""
        return "甲" * 500

    @tool
    async def query_product(product_name: str) -> str:
        """查商品。"""
        return "乙" * 500

    monkeypatch.setattr(
        agent_module, "build_tools", lambda *args, **kwargs: [query_order, query_product]
    )
    calls = [
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1", "type": "tool_call"},
        {"name": "query_product", "args": {"product_name": "猫粮"}, "id": "c2", "type": "tool_call"},
    ]
    state = _state("business", None, calls)
    runtime = _runtime(tool_result_max_tokens=100)
    update = await agent_module.agent_tools(state, runtime)
    total = runtime.context.counter.count_messages(update["messages"])
    assert total <= 100
    assert all("工具结果已截断" in m.content for m in update["messages"])
