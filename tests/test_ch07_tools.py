from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.graph.agent as agent_module
from app.services.token_counter import TokenCounter
from app.tools.builtin import register_builtins
from app.tools.contracts import ToolDefinition, ToolOutcome
from app.tools.registry import ToolRegistry, allowed_tool_names


def test_tool_order_is_static():
    registry = ToolRegistry()
    register_builtins(registry)
    assert list(registry.snapshot()) == [
        "query_order", "query_product", "query_faq", "submit_refund", "create_ticket",
    ]


def test_refund_whitelist_only_allows_refund_and_ticket():
    assert allowed_tool_names("refund_flow", "refund_return") == {
        "submit_refund", "create_ticket",
    }
    assert allowed_tool_names("refund_flow", "after_sale") == {"create_ticket"}
    assert "query_faq" not in allowed_tool_names("knowledge", None)


def _runtime(tool_result_max_tokens=1200, registry=None):
    return SimpleNamespace(
        context=SimpleNamespace(
            session_factory=None,
            token_budget=4096,
            tool_timeout=5.0,
            tool_retries=1,
            turn=SimpleNamespace(citations=[]),
            counter=TokenCounter(estimate_factor=1.0),
            tool_result_max_tokens=tool_result_max_tokens,
            registry=registry,
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
        "citations": [],
        "pending_tool_calls": list(calls),
        "tool_cursor": 0,
        "tool_tokens_used": 0,
    }


@pytest.mark.asyncio
async def test_blocked_submit_refund_never_creates_form():
    # 业务路由下没有已确认订单：submit_refund 专属守卫（订单一致 + 退款资格）拦下，不产出表单。
    state = _state("business", "refund_return", [{
        "name": "submit_refund", "args": {"order_no": "1001"},
        "id": "c1", "type": "tool_call",
    }])
    update = await agent_module.agent_tools(state, _runtime())
    assert update["suggested_actions"] == []
    assert update["refund_offered"] is False
    assert "不能生成表单" in update["messages"][0].content


@pytest.mark.asyncio
async def test_blocked_create_ticket_never_adds_actions():
    # 知识路由下误调 submit_refund（同样没有已确认订单）也被守卫拦下，不产生任何动作。
    state = _state("knowledge", None, [{
        "name": "submit_refund", "args": {"order_no": "1001"},
        "id": "c1", "type": "tool_call",
    }])
    update = await agent_module.agent_tools(state, _runtime())
    assert update["suggested_actions"] == []


def _long_probe(text):
    async def invoke(arguments, context):
        return ToolOutcome("success", data=None, message=text)
    return invoke


@pytest.mark.asyncio
async def test_tool_result_total_is_capped():
    registry = ToolRegistry()
    register_builtins(registry)
    registry.register(ToolDefinition(
        name="probe_a", description="探针A", source="builtin", server=None,
        original_name=None, invoke=_long_probe("甲" * 500),
        parameters={"type": "object", "properties": {}, "required": []},
    ))
    registry.register(ToolDefinition(
        name="probe_b", description="探针B", source="builtin", server=None,
        original_name=None, invoke=_long_probe("乙" * 500),
        parameters={"type": "object", "properties": {}, "required": []},
    ))
    calls = [
        {"name": "probe_a", "args": {}, "id": "c1", "type": "tool_call"},
        {"name": "probe_b", "args": {}, "id": "c2", "type": "tool_call"},
    ]
    state = _state("business", None, calls)
    runtime = _runtime(tool_result_max_tokens=100, registry=registry)

    first = await agent_module.agent_tools(state, runtime)
    assert "工具结果已截断" in first["messages"][0].content

    state["messages"] = state["messages"] + first["messages"]
    state["tool_cursor"] = first["tool_cursor"]
    state["tool_tokens_used"] = first["tool_tokens_used"]
    second = await agent_module.agent_tools(state, runtime)
    assert "工具结果已截断" in second["messages"][0].content

    total = runtime.context.counter.count_messages(first["messages"] + second["messages"])
    assert total <= 100
