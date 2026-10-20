import json
import logging
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.graph.nodes as nodes
from app.db.models import LowConfidenceQuestion
from app.db.repository import ChatRepository
from app.graph.routing import gate_router
from app.rag.pipeline import RagResult, TurnContext
from app.prompts.orchestrator import (
    AGENT_STOPPED_REPLY,
    CHITCHAT_REPLY,
    COMPLAINT_REPLY,
    KNOWLEDGE_UNAVAILABLE_REPLY,
)


def _runtime(session_factory=None, raw_question="退货政策"):
    return SimpleNamespace(
        context=SimpleNamespace(
            session_factory=session_factory,
            turn=TurnContext(raw_question=raw_question, conversation_id=1),
        )
    )


def _state(**overrides):
    state = {
        "messages": [HumanMessage(content="退货政策是什么")],
        "conversation_id": 1,
        "trace": {"t0": 0.0},
        "steps": 0,
        "tokens_used": 0,
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_coref_is_passthrough():
    assert await nodes.coref(_state(), _runtime()) == {}


@pytest.mark.asyncio
async def test_forced_rag_strong(monkeypatch):
    result = RagResult(
        refused=False, raw_query="退货政策", strategy="hybrid_rerank",
        evidence_text="[1] 7 天无理由退货", citations=[{"n": 1}], rerank_score=0.71,
    )
    seen = {}

    def fake_pipeline(question, **kwargs):
        seen["question"] = question
        seen["kwargs"] = kwargs
        return result

    monkeypatch.setattr(nodes, "run_pipeline", fake_pipeline)
    update = await nodes.forced_rag(_state(), _runtime())
    assert update["evidence"] == "[1] 7 天无理由退货"
    assert update["citations"] == [{"n": 1}]
    assert update["evidence_strong"] is True
    assert update["retrieval_error"] is None
    assert update["trace"]["forced_rag"] is True
    assert update["trace"]["rerank_score"] == 0.71
    assert seen["question"] == "退货政策是什么"


@pytest.mark.asyncio
async def test_forced_rag_weak_keeps_refusal_fields(monkeypatch):
    result = RagResult(
        refused=True, raw_query="退货政策", strategy="hybrid_rerank",
        refusal_source="self_check", refusal_reason="资料没写拆封后退货",
    )
    monkeypatch.setattr(nodes, "run_pipeline", lambda *a, **kw: result)
    update = await nodes.forced_rag(_state(), _runtime())
    assert update["evidence_strong"] is False
    assert update["refusal_source"] == "self_check"
    assert update["refusal_reason"] == "资料没写拆封后退货"


@pytest.mark.asyncio
async def test_forced_rag_unavailable_is_not_weak_evidence(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(nodes, "run_pipeline", boom)
    update = await nodes.forced_rag(_state(), _runtime())
    assert update["evidence_strong"] is False
    assert update["retrieval_error"] == "milvus down"
    assert update["trace"]["retrieval_error"] == "milvus down"


@pytest.mark.asyncio
async def test_confidence_check_sets_reason_and_stop():
    weak = await nodes.confidence_check(
        _state(evidence_strong=False), _runtime()
    )
    assert weak["fallback_reason"] == "evidence_weak"
    assert weak["stop_reason"] == "gate_refused"

    down = await nodes.confidence_check(
        _state(evidence_strong=False, retrieval_error="milvus down"), _runtime()
    )
    assert down["fallback_reason"] == "retrieval_unavailable"
    assert down["stop_reason"] == "retrieval_unavailable"

    ok = await nodes.confidence_check(_state(evidence_strong=True), _runtime())
    assert "fallback_reason" not in ok


def test_gate_router():
    assert gate_router({"evidence_strong": True, "retrieval_error": None}) == "agent"
    assert gate_router({"evidence_strong": False, "retrieval_error": None}) == "fallback"
    assert gate_router({"evidence_strong": False, "retrieval_error": "down"}) == "fallback"


@pytest.mark.asyncio
async def test_fallback_evidence_weak_records_low_confidence(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    state = _state(
        conversation_id=cid,
        evidence_strong=False,
        fallback_reason="evidence_weak",
        stop_reason="gate_refused",
        refusal_source="self_check",
        refusal_reason="资料不够",
    )
    update = await nodes.fallback_reply(state, _runtime(session_factory))
    assert "暂时没有查到" in update["answer"]
    assert update["stop_reason"] == "gate_refused"
    async with session_factory() as session:
        rows = (await session.execute(__import__("sqlalchemy").select(LowConfidenceQuestion))).scalars().all()
    assert len(rows) == 1
    assert rows[0].raw_question == "退货政策是什么"
    assert rows[0].source.value == "self_check"


@pytest.mark.asyncio
async def test_fallback_unavailable_does_not_record(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    state = _state(
        conversation_id=cid,
        evidence_strong=False,
        retrieval_error="milvus down",
        fallback_reason="retrieval_unavailable",
        stop_reason="retrieval_unavailable",
    )
    update = await nodes.fallback_reply(state, _runtime(session_factory))
    assert update["answer"] == KNOWLEDGE_UNAVAILABLE_REPLY
    async with session_factory() as session:
        rows = (await session.execute(__import__("sqlalchemy").select(LowConfidenceQuestion))).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_fallback_agent_stopped_uses_max_steps_text():
    update = await nodes.fallback_reply(
        _state(fallback_reason=None, stop_reason=None), _runtime()
    )
    assert update["answer"] == AGENT_STOPPED_REPLY
    assert update["stop_reason"] == "max_steps"


@pytest.mark.asyncio
async def test_complaint_reply_has_two_independent_actions():
    update = await nodes.complaint_reply(
        _state(messages=[HumanMessage(content="我要投诉")]), _runtime()
    )
    assert update["answer"] == COMPLAINT_REPLY
    assert update["stop_reason"] == "fixed_reply"
    actions = update["suggested_actions"]
    assert [a["type"] for a in actions] == ["transfer_human", "create_ticket"]
    assert actions[1]["draft"]["description"] == "我要投诉"
    assert actions[1]["draft"]["ticket_type"] == "投诉"


@pytest.mark.asyncio
async def test_chitchat_reply_has_no_actions():
    update = await nodes.chitchat_reply(
        _state(messages=[HumanMessage(content="你好")]), _runtime()
    )
    assert update["answer"] == CHITCHAT_REPLY
    assert update["suggested_actions"] == []
    assert update["stop_reason"] == "fixed_reply"


@pytest.mark.asyncio
async def test_log_writes_json_trace(caplog):
    caplog.set_level(logging.INFO, logger="knowweave.turn")
    state = _state(
        intent="refund_return",
        route="knowledge",
        evidence_strong=True,
        stop_reason="no_tool_calls",
        messages=[
            HumanMessage(content="退货政策是什么"),
            AIMessage(content="", tool_calls=[
                {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}
            ]),
            AIMessage(content="答"),
        ],
        trace={"t0": 0.0, "forced_rag": True, "rerank_score": 0.71},
    )
    update = await nodes.log_turn(state, _runtime())
    payload = json.loads(caplog.records[-1].message)
    assert payload["forced_rag"] is True
    assert payload["intent"] == "refund_return"
    assert payload["tools"] == ["query_order"]
    assert payload["stop_reason"] == "no_tool_calls"
    assert update["trace"]["tools"] == ["query_order"]
