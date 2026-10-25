"""回归测试：真实 LangGraph 往返下 citations 快照跨轮保留。

ch09 现有测试用 FakeGraph，不建模 graph_input 的 seed 与 checkpoint 持久化，
导致 `citations_by_message_id`（LastValue channel）被每轮 graph_input 里的
`{}` 覆盖、上一轮快照丢失也测不出来。这里用真实编译图 + InMemorySaver 复现
ChatService.stream 的完整往返，验证快照 map 的跨轮累积与失败轮保留。
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.runtime import thread_config
from app.graph.state import ConversationState
from app.services.chat_service import ChatService


def _last_human_text(messages) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content
    return ""


class _Controller:
    def __init__(self):
        self.fail_reply = False


def _build_snapshot_graph(controller: _Controller | None = None):
    """真实编译图：reply 节点每轮产出一条消息，log 节点满足 aupdate_state 的
    as_node 回写通道。"""
    controller = controller or _Controller()

    async def reply(state: ConversationState):
        if controller.fail_reply:
            raise RuntimeError("boom")
        text = _last_human_text(state.get("messages"))
        return {
            "messages": [AIMessage(content=f"答-{text}")],
            "answer": f"答-{text}",
            "route": "business",
            "intent": "faq",
            "citations": [{"n": 1, "answer": f"片段-{text}"}],
            "suggested_actions": [],
        }

    async def log(state: ConversationState):
        return {"trace": dict(state.get("trace") or {})}

    builder = StateGraph(ConversationState)
    builder.add_node("reply", reply)
    builder.add_node("log", log)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", "log")
    builder.add_edge("log", END)
    return builder.compile(checkpointer=InMemorySaver())


async def _service(session_factory, graph):
    return ChatService(
        model=object(), intent_model=object(), session_factory=session_factory,
        graph=graph,
    )


async def _run_turn(service, session_id, message):
    events = [event async for event in service.stream(session_id, message)]
    done = next(event for event in events if event["type"] == "done")
    return events, done


async def test_real_graph_preserves_prior_turn_snapshot(session_factory):
    graph = _build_snapshot_graph()
    service = await _service(session_factory, graph)

    events1, done1 = await _run_turn(service, None, "问一")
    session_event = next(e for e in events1 if e["type"] == "session")
    conversation_id = int(session_event["session_id"])
    message1 = done1["message_id"]

    _, done2 = await _run_turn(service, str(conversation_id), "问二")
    message2 = done2["message_id"]

    # 最新一轮快照指向第 2 轮。
    latest, snapshot = await service.latest_turn_snapshot(conversation_id)
    assert latest == message2
    assert snapshot == [{"n": 1, "answer": "片段-问二"}]

    # 持久化的快照 map 同时保留第 1 轮与第 2 轮，第 1 轮不被新轮 seed 覆盖。
    persisted = dict(
        (await graph.aget_state(thread_config(conversation_id))).values or {}
    ).get("citations_by_message_id") or {}
    assert str(message1) in persisted
    assert str(message2) in persisted
    assert persisted[str(message1)] == [{"n": 1, "answer": "片段-问一"}]


async def test_real_graph_failed_turn_keeps_prior_snapshot(session_factory):
    controller = _Controller()
    graph = _build_snapshot_graph(controller)
    service = await _service(session_factory, graph)

    events1, done1 = await _run_turn(service, None, "问一")
    conversation_id = int(
        next(e for e in events1 if e["type"] == "session")["session_id"]
    )
    message1 = done1["message_id"]

    controller.fail_reply = True
    events2 = [event async for event in service.stream(str(conversation_id), "问二")]
    assert any(event["type"] == "error" for event in events2)

    # 失败的轮次不落新助手消息，也不覆盖上一轮快照。
    latest, snapshot = await service.latest_turn_snapshot(conversation_id)
    assert latest == message1
    assert snapshot == [{"n": 1, "answer": "片段-问一"}]
