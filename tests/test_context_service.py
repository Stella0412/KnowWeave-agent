from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.services.context_service import ContextStore


def test_count_tokens_monotonic():
    store = ContextStore()
    short = store.count_tokens([HumanMessage(content="你好")])
    long = store.count_tokens([HumanMessage(content="你好" * 100)])
    assert short > 0
    assert long > short


def test_trim_drops_oldest_keeps_newest():
    store = ContextStore(token_budget=60)
    system = SystemMessage(content="你是客服")
    history = [AIMessage(content="第" + str(i) + "条很长的历史消息内容") for i in range(10)]
    history.append(HumanMessage(content="最新问题"))
    trimmed = store.trim(system, history)
    # 最新一条用户消息必须保留
    assert trimmed[-1].content == "最新问题"
    # system + trimmed 不超预算
    assert store.count_tokens([system] + trimmed) <= 60


def test_trim_never_leaves_orphan_tool_message():
    # 按 token 逐条从头裁时，可能把带 tool_calls 的 AIMessage 裁掉、却留下它的 ToolMessage。
    # 上游会直接 400：Messages with role 'tool' must be a response to a preceding
    # message with 'tool_calls'。裁剪结果必须保证 tool 消息仍能找到发起它的 assistant。
    system = SystemMessage(content="sys")
    old = HumanMessage(content="旧问题" * 50)
    ai = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_1",
            "name": "query_order",
            "args": {"order_no": "1001"},
            "type": "tool_call",
        }],
    )
    tool = ToolMessage(content="订单1001已签收", tool_call_id="call_1")
    latest = HumanMessage(content="最新问题")
    budget = ContextStore().count_tokens([system, tool, latest])
    store = ContextStore(token_budget=budget)

    trimmed = store.trim(system, [old, ai, tool, latest])

    assert not isinstance(trimmed[0], ToolMessage), "裁剪后不能以孤立的 tool 消息开头"
    assert trimmed[-1].content == "最新问题"
    pending: set[str] = set()
    for message in trimmed:
        if isinstance(message, AIMessage):
            pending = {call["id"] for call in message.tool_calls}
        elif isinstance(message, ToolMessage):
            assert message.tool_call_id in pending
    assert store.count_tokens([system] + trimmed) <= budget


def test_trim_does_not_mutate_input():
    store = ContextStore(token_budget=1000)
    history = [HumanMessage(content="你好")]
    store.trim(SystemMessage(content="sys"), history)
    assert len(history) == 1
