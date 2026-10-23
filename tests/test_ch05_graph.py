from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import app.graph.nodes as nodes
from app.db.repository import ChatRepository
from app.graph.build import build_graph
from app.graph.state import GraphContext
from app.prompts.orchestrator import (
    AGENT_STOPPED_REPLY,
    CHITCHAT_REPLY,
    COMPLAINT_REPLY,
    KNOWLEDGE_UNAVAILABLE_REPLY,
    IntentOutput,
)
from app.rag.pipeline import RagResult, TurnContext


class FakeIntentModel:
    def __init__(self, intent):
        self.intent = intent

    async def ainvoke(self, messages):
        return IntentOutput(intent=self.intent, confidence=0.9)


class ScriptedAgentModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.seen = []

    def bind_tools(self, tools):
        self.bound_tools = [t["name"] for t in tools]
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.seen.append(list(messages))
        return self.responses.pop(0)


def _input(question, **overrides):
    state = {
        "messages": [HumanMessage(content=question)],
        "user_id": "anonymous",
        "conversation_id": 1,
        "intent": None,
        "intent_source": None,
        "route": None,
        "evidence": None,
        "citations": [],
        "evidence_strong": None,
        "retrieval_error": None,
        "refusal_source": None,
        "refusal_reason": None,
        "answer": None,
        "steps": 0,
        "tokens_used": 0,
        "suggested_actions": [],
        "fallback_reason": None,
        "stop_reason": None,
        "trace": {"t0": 0.0},
    }
    state.update(overrides)
    return state


def _context(intent, agent_model, session_factory):
    return GraphContext(
        model=agent_model,
        intent_model=FakeIntentModel(intent),
        session_factory=session_factory,
        turn=TurnContext(raw_question="测试", conversation_id=1),
    )


async def _invoke(graph, state, context, thread_id="1"):
    return await graph.ainvoke(
        state,
        config={"configurable": {"thread_id": thread_id}},
        context=context,
    )


def test_runtime_thread_config_uses_string_thread_id():
    from app.graph.runtime import thread_config

    assert thread_config(12) == {"configurable": {"thread_id": "12"}}


def test_normalize_stream_event_accepts_both_shapes():
    from app.graph.runtime import normalize_stream_event

    assert normalize_stream_event(("messages", {"x": 1})) == ("messages", {"x": 1})
    assert normalize_stream_event({"type": "updates", "data": {"y": 2}}) == ("updates", {"y": 2})
    with pytest.raises(ValueError):
        normalize_stream_event({"unexpected": True})


@pytest.mark.asyncio
async def test_kg_question_forced_rag_then_agent(monkeypatch, session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()

    monkeypatch.setattr(
        nodes, "run_pipeline",
        lambda *a, **kw: RagResult(
            refused=False, raw_query="退货政策", strategy="hybrid_rerank",
            evidence_text="[1] 7 天无理由退货", citations=[{"n": 1}], rerank_score=0.7,
        ),
    )
    agent = ScriptedAgentModel([AIMessage(content="按证据回答")])
    graph = build_graph(checkpointer=InMemorySaver())
    state = await _invoke(
        graph, _input("退货政策是什么", conversation_id=cid),
        _context("product_consult", agent, session_factory),
    )
    assert state["route"] == "knowledge"
    assert state["trace"]["forced_rag"] is True
    assert state["answer"] == "按证据回答"
    assert agent.calls == 1
    assert "7 天无理由退货" in agent.seen[0][-1].content
    assert agent.bound_tools == [
        "query_order", "query_product",
        "query_faq", "submit_refund", "create_ticket",
    ]


@pytest.mark.asyncio
async def test_business_question_skips_rag_and_takes_two_steps(monkeypatch, session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()

    def no_rag(*args, **kwargs):
        raise AssertionError("业务路不应调用 run_pipeline")

    monkeypatch.setattr(nodes, "run_pipeline", no_rag)
    agent = ScriptedAgentModel([
        AIMessage(content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}
        ]),
        AIMessage(content="", tool_calls=[
            {"name": "logistics_query_logistics", "args": {"tracking_no": "SF1001CN"}, "id": "c2", "type": "tool_call"}
        ]),
        AIMessage(content="已为您查到物流"),
    ])
    graph = build_graph(checkpointer=InMemorySaver())
    state = await _invoke(
        graph, _input("订单 1001 的物流到哪了", conversation_id=cid),
        _context("logistics", agent, session_factory),
    )
    assert state["route"] == "business"
    assert state["steps"] == 3
    assert state["trace"]["tools"] == ["query_order", "logistics_query_logistics"]
    assert state["answer"] == "已为您查到物流"


@pytest.mark.asyncio
async def test_complaint_never_calls_agent(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()

    class BoomModel:
        def bind_tools(self, tools):
            raise AssertionError("投诉不应进 Agent")

        async def ainvoke(self, messages):
            raise AssertionError("投诉不应进 Agent")

    graph = build_graph(checkpointer=InMemorySaver())
    state = await _invoke(
        graph, _input("我要投诉", conversation_id=cid),
        _context("complaint", BoomModel(), session_factory),
    )
    assert state["answer"] == COMPLAINT_REPLY
    assert [a["type"] for a in state["suggested_actions"]] == [
        "transfer_human", "create_ticket",
    ]


@pytest.mark.asyncio
async def test_chitchat_never_calls_agent(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()

    class BoomModel:
        def bind_tools(self, tools):
            raise AssertionError("闲聊不应进 Agent")

        async def ainvoke(self, messages):
            raise AssertionError("闲聊不应进 Agent")

    graph = build_graph(checkpointer=InMemorySaver())
    state = await _invoke(
        graph, _input("你好", conversation_id=cid),
        _context("chitchat", BoomModel(), session_factory),
    )
    assert state["answer"] == CHITCHAT_REPLY
    assert state["suggested_actions"] == []


@pytest.mark.asyncio
async def test_weak_evidence_blocks_agent_and_records_pool(monkeypatch, session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()

    monkeypatch.setattr(
        nodes, "run_pipeline",
        lambda *a, **kw: RagResult(
            refused=True, raw_query="退货政策", strategy="hybrid_rerank",
            refusal_source="self_check", refusal_reason="资料不够",
        ),
    )

    class BoomModel:
        def bind_tools(self, tools):
            raise AssertionError("弱证据不应进 Agent")

        async def ainvoke(self, messages):
            raise AssertionError("弱证据不应进 Agent")

    graph = build_graph(checkpointer=InMemorySaver())
    state = await _invoke(
        graph, _input("拆封了还能退吗", conversation_id=cid),
        _context("product_consult", BoomModel(), session_factory),
    )
    assert "暂时没有查到" in state["answer"]
    from sqlalchemy import select
    from app.db.models import LowConfidenceQuestion
    async with session_factory() as session:
        rows = (await session.execute(select(LowConfidenceQuestion))).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_retrieval_unavailable_blocks_agent_without_pool(monkeypatch, session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()

    monkeypatch.setattr(nodes, "run_pipeline", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("down")))

    class BoomModel:
        def bind_tools(self, tools):
            raise AssertionError("知识库不可用不应进 Agent")

        async def ainvoke(self, messages):
            raise AssertionError("知识库不可用不应进 Agent")

    graph = build_graph(checkpointer=InMemorySaver())
    state = await _invoke(
        graph, _input("退货政策是什么", conversation_id=cid),
        _context("product_consult", BoomModel(), session_factory),
    )
    assert state["answer"] == KNOWLEDGE_UNAVAILABLE_REPLY
    from sqlalchemy import select
    from app.db.models import LowConfidenceQuestion
    async with session_factory() as session:
        rows = (await session.execute(select(LowConfidenceQuestion))).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_checkpoint_carries_second_turn_and_resets_counters(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()

    agent = ScriptedAgentModel([
        AIMessage(content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}
        ]),
        AIMessage(content="第一轮"),
        AIMessage(content="第二轮"),
    ])
    graph = build_graph(checkpointer=InMemorySaver())
    ctx = _context("logistics", agent, session_factory)
    first = await _invoke(graph, _input("订单 1001", conversation_id=cid), ctx)
    assert first["steps"] == 2

    second = await _invoke(
        graph, _input("你好", conversation_id=cid, steps=0, tokens_used=0, trace={"t0": 0.0}),
        _context("chitchat", agent, session_factory),
    )
    assert len(second["messages"]) == 6
    assert second["steps"] == 0
    assert second["answer"] == CHITCHAT_REPLY


@pytest.mark.asyncio
async def test_max_steps_forces_fallback_without_executing_tool(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()

    agent = ScriptedAgentModel([
        AIMessage(content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}
        ]),
    ])
    graph = build_graph(checkpointer=InMemorySaver(), max_agent_steps=1)
    state = await _invoke(
        graph, _input("订单 1001", conversation_id=cid),
        _context("logistics", agent, session_factory),
    )
    assert state["answer"] == AGENT_STOPPED_REPLY
    assert state["stop_reason"] == "max_steps"
    assert not any(type(m).__name__ == "ToolMessage" for m in state["messages"])


@pytest.mark.asyncio
async def test_create_ticket_intercept_end_to_end(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()

    agent = ScriptedAgentModel([
        AIMessage(content="", tool_calls=[{
            "name": "create_ticket",
            "args": {"description": "商品破损", "ticket_type": "售后"},
            "id": "c1", "type": "tool_call",
        }]),
        AIMessage(content="已为您展示建单选项"),
    ])
    graph = build_graph(checkpointer=InMemorySaver())
    state = await _invoke(
        graph, _input("商品破损", conversation_id=cid),
        _context("clarify", agent, session_factory),
    )
    assert state["suggested_actions"][0]["type"] == "create_ticket"
    assert state["answer"] == "已为您展示建单选项"
    from sqlalchemy import select
    from app.db.models import Ticket
    async with session_factory() as session:
        assert (await session.execute(select(Ticket))).scalars().all() == []
