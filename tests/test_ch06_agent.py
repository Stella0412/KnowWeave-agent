from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph.agent import agent_llm, agent_tools
from app.prompts.context import build_dynamic_block
from app.services.token_counter import TokenCounter
from app.tools.business import REFUND_REASON_CATEGORIES


class FakeAgent:
    def __init__(self):
        self.bound = []
        self.calls = []

    def bind_tools(self, tools):
        self.bound = [t["name"] for t in tools]
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="答")


@pytest.mark.asyncio
@pytest.mark.parametrize("order_no", ["1004", "1005"])
async def test_refund_keeps_layered_history_but_current_order_is_authoritative(
    session_factory, order_no,
):
    model = FakeAgent()
    runtime = _runtime(session_factory)
    runtime.context.model = model
    old = [
        HumanMessage(content="我要退1001", id="db:1"),
        AIMessage(content="订单1001猫爬架可以退"),
    ]
    current = [HumanMessage(content="我要退款", id="db:2")]
    state = _state(
        messages=old + current,
        resolved_query="我要退款",
        order_no=order_no,
        order_data={"order_no": order_no, "product_name": "猫砂" if order_no == "1004" else "猫罐头"},
    )
    await agent_llm(state, runtime)
    sent = model.calls[0]
    assert any("1001" in str(m.content) for m in sent)  # 历史仍作背景
    user = next(m for m in reversed(sent) if isinstance(m, HumanMessage))
    assert order_no in user.content
    assert "【判定优先级】" in user.content
    assert state["messages"] == old + current
    assert current[0].content == "我要退款"


@pytest.mark.asyncio
@pytest.mark.parametrize("order_no", ["1004", "1005"])
async def test_refund_second_inference_keeps_current_tool_pairing(
    session_factory, order_no,
):
    model = FakeAgent()
    runtime = _runtime(session_factory)
    runtime.context.model = model
    old = [
        HumanMessage(content="我要退1001", id="db:1"),
        AIMessage(content="订单1001猫爬架可以退"),
    ]
    current = [
        HumanMessage(content="我要退款", id="db:2"),
        AIMessage(content="", tool_calls=[{
            "name": "submit_refund", "args": {"order_no": order_no},
            "id": "current-call", "type": "tool_call",
        }]),
        ToolMessage(content="该订单不能生成退款表单", tool_call_id="current-call"),
    ]
    state = _state(
        messages=old + current,
        resolved_query="我要退款",
        order_no=order_no,
        order_data={"order_no": order_no, "product_name": "猫砂"},
    )
    await agent_llm(state, runtime)
    sent = model.calls[0]
    assert sent[-1].tool_call_id == "current-call"
    assert sent[-2].tool_calls[0]["id"] == "current-call"


def _runtime(session_factory):
    return SimpleNamespace(
        context=SimpleNamespace(
            session_factory=session_factory,
            token_budget=4096,
            tool_timeout=5.0,
            tool_retries=1,
            turn=SimpleNamespace(citations=[]),
            counter=TokenCounter(estimate_factor=1.0),
            tool_result_max_tokens=1200,
            context_manager=None,
            budget=None,
            settings=None,
        )
    )


def _state(**overrides):
    state = {
        "messages": [],
        "conversation_id": 1,
        "intent": "refund_return",
        "route": "refund_flow",
        "order_no": "1001",
        "order_data": {
            "order_no": "1001",
            "product_name": "猫爬架",
            "status": "已签收",
            "product_type": "普通",
            "signed_at": (datetime.now() - timedelta(days=3)).isoformat(),
        },
        "evidence": "[1] 七天无理由",
        "steps": 0,
        "tokens_used": 0,
        "suggested_actions": [],
        "citations": [],
        "pending_tool_calls": [],
        "tool_cursor": 0,
        "tool_tokens_used": 0,
        "refund_offered": None,
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_refund_agent_binds_full_fixed_tool_set(session_factory):
    model = FakeAgent()
    runtime = _runtime(session_factory)
    runtime.context.model = model
    await agent_llm(_state(), runtime)
    assert model.bound == [
        "query_order", "query_product",
        "query_faq", "submit_refund", "create_ticket",
    ]


def test_refund_dynamic_block_contains_order_and_policy():
    text = build_dynamic_block(
        route="refund_flow", intent="refund_return",
        summary="", evidence="[1] 七天无理由",
        order_data={"order_no": "1001", "product_name": "猫爬架"},
        counter=TokenCounter(estimate_factor=1.0),
        evidence_budget_tokens=1000, order_max_tokens=600,
    )
    assert "1001" in text
    assert "猫爬架" in text
    assert "七天无理由" in text
    assert "不要追问退款原因" in text
    assert "普通商品" in text
    assert "定制类、生鲜类" in text


def test_refund_dynamic_block_requires_fresh_form_on_later_request():
    # 上一轮已经生成过表单时，模型会读到历史里的「不要再次调用 submit_refund」，
    # 下一轮用户再说「我要退款」就只复述「表单刚才已生成」，前端拿不到新的表单动作。
    text = build_dynamic_block(
        route="refund_flow", intent="refund_return",
        summary="", evidence="[1] 七天无理由",
        order_data={"order_no": "1001", "product_name": "猫爬架"},
        counter=TokenCounter(estimate_factor=1.0),
        evidence_budget_tokens=1000, order_max_tokens=600,
    )
    assert "不会留到本轮" in text
    assert "再次要求退款" in text
    assert "必须重新调用 submit_refund" in text


def test_logistics_dynamic_block_forbids_refund_arrival_reading():
    # 「那大概什么时候能到」紧跟在退款答复后，但本轮 intent 已被消解成物流；
    # 通用 business 提示不够，模型会答成「退款到账时间」。
    text = build_dynamic_block(
        route="business", intent="logistics",
        summary="", evidence=None, order_data=None,
        counter=TokenCounter(estimate_factor=1.0),
        evidence_budget_tokens=1000, order_max_tokens=600,
    )
    assert "物流查询" in text
    assert "不是退款到账时间" in text
    assert "重新调用 logistics_query_logistics" in text


@pytest.mark.asyncio
async def test_agent_llm_sends_resolved_query_instead_of_ambiguous_followup(session_factory):
    # 指代消解在 coref 节点做完了，agent 看到的却还是原话「那大概什么时候能到」，
    # 紧邻的退款回答会把「到」带偏。发给模型的本轮用户消息必须是消解后的完整问句。
    model = FakeAgent()
    runtime = _runtime(session_factory)
    runtime.context.model = model
    raw = HumanMessage(content="那大概什么时候能到")
    state = _state(
        intent="logistics",
        route="business",
        resolved_query="订单1001的物流单号SF1001CN大概什么时候能到",
        order_data=None,
        evidence=None,
        messages=[raw],
    )
    await agent_llm(state, runtime)
    assert model.calls[0][-1].content.startswith("订单1001的物流单号SF1001CN大概什么时候能到")
    # 物流路由说明进入动态块，仍然禁止把「到」读成退款到账。
    assert "不是退款到账时间" in model.calls[0][-1].content
    # 只改发给模型的副本，状态/落库里的用户原话不能被改写
    assert raw.content == "那大概什么时候能到"


def test_after_sale_dynamic_block_uses_after_sale_addendum():
    text = build_dynamic_block(
        route="refund_flow", intent="after_sale",
        summary="", evidence="[1] 保修条款",
        order_data={"order_no": "1001"},
        counter=TokenCounter(estimate_factor=1.0),
        evidence_budget_tokens=1000, order_max_tokens=600,
    )
    assert "售后判定" in text
    assert "不要调用 submit_refund" in text


@pytest.mark.asyncio
async def test_after_sale_agent_binds_full_fixed_tool_set(session_factory):
    model = FakeAgent()
    runtime = _runtime(session_factory)
    runtime.context.model = model
    await agent_llm(_state(intent="after_sale"), runtime)
    assert model.bound == [
        "query_order", "query_product",
        "query_faq", "submit_refund", "create_ticket",
    ]


@pytest.mark.asyncio
async def test_after_sale_create_ticket_without_evidence_is_denied(session_factory):
    state = _state(
        intent="after_sale",
        messages=[AIMessage(content="", tool_calls=[{
            "name": "create_ticket",
            "args": {"description": "猫爬架破损", "ticket_type": "售后"},
            "id": "c1", "type": "tool_call",
        }])],
        pending_tool_calls=[{
            "name": "create_ticket",
            "args": {"description": "猫爬架破损", "ticket_type": "售后"},
            "id": "c1", "type": "tool_call",
        }],
        tool_cursor=0,
    )
    update = await agent_tools(state, _runtime(session_factory))
    # 新语义：after_sale 不再注入 transfer_human/create_ticket 动作；create_ticket 只走引擎，
    # 且没有服务端建单证据时被守卫拦截。
    types = [item["type"] for item in update["suggested_actions"]]
    assert types == []
    assert "transfer_human" not in types
    assert "没有有效的用户建单请求证据" in update["messages"][0].content


@pytest.mark.asyncio
async def test_submit_refund_intercept_adds_form(session_factory):
    state = _state(
        messages=[AIMessage(content="", tool_calls=[{
            "name": "submit_refund", "args": {"order_no": "1001"},
            "id": "c1", "type": "tool_call",
        }])],
        pending_tool_calls=[{
            "name": "submit_refund", "args": {"order_no": "1001"},
            "id": "c1", "type": "tool_call",
        }],
        tool_cursor=0,
    )
    update = await agent_tools(state, _runtime(session_factory))
    action = update["suggested_actions"][0]
    assert action["type"] == "refund_form"
    assert action["draft"]["order_no"] == "1001"
    assert tuple(action["draft"]["reason_options"]) == REFUND_REASON_CATEGORIES
    assert update["refund_offered"] is True


@pytest.mark.asyncio
async def test_submit_refund_intercept_blocks_ineligible_order(session_factory):
    # 模型偶尔会误判；定制类订单即使模型调了 submit_refund，也不能出现可填写的表单。
    signed = (datetime.now() - timedelta(days=3)).isoformat()
    state = _state(
        order_no="1002",
        order_data={
            "order_no": "1002",
            "product_name": "定制猫窝",
            "status": "已签收",
            "product_type": "定制",
            "signed_at": signed,
        },
        messages=[AIMessage(content="", tool_calls=[{
            "name": "submit_refund", "args": {"order_no": "1002"},
            "id": "c1", "type": "tool_call",
        }])],
        pending_tool_calls=[{
            "name": "submit_refund", "args": {"order_no": "1002"},
            "id": "c1", "type": "tool_call",
        }],
        tool_cursor=0,
    )
    update = await agent_tools(state, _runtime(session_factory))
    assert update["suggested_actions"] == []
    assert update["refund_offered"] is False
    assert "定制" in update["messages"][0].content


@pytest.mark.asyncio
async def test_refund_tool_message_scopes_repeat_call_to_current_turn(session_factory):
    state = _state(
        messages=[AIMessage(content="", tool_calls=[{
            "name": "submit_refund", "args": {"order_no": "1001"},
            "id": "c1", "type": "tool_call",
        }])],
        pending_tool_calls=[{
            "name": "submit_refund", "args": {"order_no": "1001"},
            "id": "c1", "type": "tool_call",
        }],
        tool_cursor=0,
    )
    update = await agent_tools(state, _runtime(session_factory))
    content = update["messages"][0].content
    # 「不要再次调用」只能约束本轮；否则这条工具消息落库后，
    # 下一轮用户再要求退款时模型会以为表单还在，不再调用 submit_refund。
    assert "本轮" in content
    assert "用户下一轮再次要求退款" in content
    assert "必须重新调用 submit_refund" in content


@pytest.mark.asyncio
async def test_submit_refund_wrong_order_does_not_add_form(session_factory):
    state = _state(
        messages=[AIMessage(content="", tool_calls=[{
            "name": "submit_refund", "args": {"order_no": "9999"},
            "id": "c1", "type": "tool_call",
        }])],
        pending_tool_calls=[{
            "name": "submit_refund", "args": {"order_no": "9999"},
            "id": "c1", "type": "tool_call",
        }],
        tool_cursor=0,
    )
    update = await agent_tools(state, _runtime(session_factory))
    assert update["suggested_actions"] == []
    assert update["refund_offered"] is False
