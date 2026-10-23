import logging
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import app.graph.agent as agent_module
from app.graph.agent import agent_llm
from app.prompts.customer_service import SYSTEM_PROMPT
from app.services.token_counter import TokenCounter


class FakeModel:
    def __init__(self):
        self.bound = []
        self.calls = []

    def bind_tools(self, tools):
        self.bound = [t["name"] for t in tools]
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="答")


class StubManager:
    def __init__(self, prefix, current, react):
        self.prefix = prefix
        self.current = current
        self.react = react

    def model_history(self, messages, meta):
        return self.prefix, self.current, self.react


def _runtime(manager=None):
    return SimpleNamespace(
        context=SimpleNamespace(
            model=FakeModel(),
            session_factory=None,
            token_budget=4096,
            tool_timeout=5.0,
            tool_retries=1,
            turn=SimpleNamespace(citations=[]),
            counter=TokenCounter(estimate_factor=1.0),
            budget=SimpleNamespace(as_dict=lambda: {"layer1": 1}),
            tool_result_max_tokens=1200,
            context_manager=manager,
        )
    )


def _meta():
    return {
        "turn_user_msg_id": 9,
        "effective_summary_upto": 1,
        "layer1_from": 2,
        "summary_text": "用户买过猫爬架",
        "summary_segments": [1],
    }


@pytest.mark.asyncio
async def test_message_order_and_only_one_system(caplog):
    current = HumanMessage(content="原话", id="db:9")
    react = [
        AIMessage(content="", tool_calls=[{
            "name": "query_order", "args": {"order_id": "1001"},
            "id": "c1", "type": "tool_call",
        }]),
        __import__("langchain_core.messages", fromlist=["ToolMessage"]).ToolMessage(
            content="订单结果", tool_call_id="c1"
        ),
    ]
    prefix = [
        HumanMessage(content="层2问题", id="db:1"),
        AIMessage(content="层2答…"),
        HumanMessage(content="层1问题", id="db:2"),
        AIMessage(content="层1答"),
    ]
    manager = StubManager(prefix, current, react)
    runtime = _runtime(manager)
    state = {
        "messages": prefix + [current] + react,
        "conversation_id": 1,
        "route": "refund_flow",
        "intent": "refund_return",
        "resolved_query": "订单1001能退吗",
        "order_no": "1001",
        "order_data": {"order_no": "1001", "product_name": "猫爬架"},
        "evidence": "[1] 七天无理由",
        "context_meta": _meta(),
        "steps": 0,
        "tokens_used": 0,
    }
    with caplog.at_level(logging.INFO, logger="knowweave.context"):
        await agent_llm(state, runtime)
    sent = runtime.context.model.calls[0]
    assert isinstance(sent[0], SystemMessage)
    assert sent[0].content == SYSTEM_PROMPT
    assert sum(isinstance(m, SystemMessage) for m in sent) == 1
    assert [m.content for m in sent[1:5]] == ["层2问题", "层2答…", "层1问题", "层1答"]
    assert sent[5].content.startswith("订单1001能退吗")
    assert "【历史梗概】" in sent[5].content
    assert "【本轮已确认订单】" in sent[5].content
    assert sent[5].content.index("【本轮已确认订单】") < sent[5].content.index("【本轮上下文资料")
    assert sent[6].tool_calls[0]["id"] == "c1"
    assert sent[7].tool_call_id == "c1"
    assert "model_ctx" in caplog.text
    assert runtime.context.model.bound == [
        "query_order", "query_product",
        "query_faq", "submit_refund", "create_ticket",
    ]


@pytest.mark.asyncio
async def test_missing_context_meta_falls_back_to_all_history(caplog):
    model = FakeModel()
    runtime = _runtime(None)
    runtime.context.model = model
    old = [HumanMessage(content="旧问", id="db:1"), AIMessage(content="旧答")]
    current = HumanMessage(content="当前", id="db:2")
    state = {
        "messages": old + [current],
        "conversation_id": 1,
        "route": "business",
        "intent": "order",
        "steps": 0,
        "tokens_used": 0,
    }
    await agent_llm(state, runtime)
    sent = model.calls[0]
    assert [m.content for m in sent[1:3]] == ["旧问", "旧答"]
    assert sent[3].content.startswith("当前")
