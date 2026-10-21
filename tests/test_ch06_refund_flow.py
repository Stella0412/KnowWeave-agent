from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.db.seed import seed_orders
from app.graph import nodes
from app.graph.routing import refund_order_router
from app.rag.pipeline import RagResult


def _runtime(session_factory, expansion_model=None):
    return SimpleNamespace(
        context=SimpleNamespace(
            session_factory=session_factory,
            expansion_model=expansion_model,
            turn=SimpleNamespace(source_files={}),
        )
    )


@pytest.mark.asyncio
async def test_fetch_order_uses_number_present_in_table(session_factory):
    await seed_orders(session_factory)
    state = {
        "resolved_query": "订单1001的猫爬架能退吗",
        "user_id": "anonymous",
        "flow_steps": [],
        "trace": {},
    }
    update = await nodes.refund_fetch_order(state, _runtime(session_factory))
    assert update["order_no"] == "1001"
    assert update["order_data"]["product_name"] == "猫爬架"
    assert update["flow_steps"] == ["fetch_order"]


@pytest.mark.asyncio
async def test_fetch_order_empty_table_returns_order_unavailable(session_factory):
    state = {
        "resolved_query": "这个能退吗",
        "user_id": "anonymous",
        "flow_steps": [],
        "trace": {},
    }
    update = await nodes.refund_fetch_order(state, _runtime(session_factory))
    assert update["order_error"] == "empty"
    assert update["fallback_reason"] == "order_unavailable"
    assert "order_no" not in update


@pytest.mark.asyncio
async def test_fetch_order_missing_number_interrupts_then_uses_choice(
    monkeypatch, session_factory
):
    await seed_orders(session_factory)
    seen = []

    def fake_interrupt(payload):
        seen.append(payload)
        return {"order_no": "1002"}

    monkeypatch.setattr(nodes, "interrupt", fake_interrupt)
    update = await nodes.refund_fetch_order(
        {
            "resolved_query": "这个能退吗",
            "user_id": "anonymous",
            "intent": "refund_return",
            "flow_steps": [],
            "trace": {},
        },
        _runtime(session_factory),
    )
    assert len(seen) == 1
    assert seen[0]["kind"] == "select_order"
    assert seen[0]["prompt"] == "请选择要退款的订单"
    assert {item["order_no"] for item in seen[0]["orders"]} == {
        "1001", "1002", "1003", "1004", "1005",
    }
    assert update["order_no"] == "1002"
    assert update["order_data"]["product_type"] == "定制"
    assert update["flow_steps"] == ["fetch_order"]


@pytest.mark.asyncio
async def test_fetch_order_resume_with_unknown_number_sets_not_found(
    monkeypatch, session_factory
):
    await seed_orders(session_factory)
    monkeypatch.setattr(nodes, "interrupt", lambda payload: {"order_no": "9999"})
    update = await nodes.refund_fetch_order(
        {
            "resolved_query": "这个能退吗",
            "user_id": "anonymous",
            "intent": "refund_return",
            "flow_steps": [],
            "trace": {},
        },
        _runtime(session_factory),
    )
    assert update["order_error"] == "not_found"
    assert update["fallback_reason"] == "order_unavailable"
    assert "order_data" not in update


def test_refund_order_router():
    assert refund_order_router({"order_error": None}) == "expand"
    assert refund_order_router({"order_error": "empty"}) == "fallback"


def test_refund_flow_edges_are_wired():
    from app.graph.build import build_graph

    edges = {
        (edge.source, edge.target)
        for edge in build_graph(checkpointer=None).get_graph().edges
    }
    assert ("classify_intent", "refund_fetch_order") in edges
    assert ("refund_fetch_order", "refund_expand") in edges
    assert ("refund_fetch_order", "fallback_reply") in edges
    assert ("refund_expand", "refund_retrieve") in edges
    assert ("refund_retrieve", "confidence_check") in edges


@pytest.mark.asyncio
async def test_retrieve_weak_policy_sets_weak_evidence(monkeypatch, session_factory):
    monkeypatch.setattr(
        nodes, "run_expanded_pipeline",
        lambda *a, **kw: RagResult(
            refused=True, raw_query="q", strategy="hybrid_rerank",
            refusal_source="retrieval_low_conf", refusal_reason="无候选",
        ),
    )
    state = {
        "resolved_query": "订单1001能退吗",
        "policy_queries": ["q1", "q2", "q3"],
        "flow_steps": [],
        "trace": {},
    }
    update = await nodes.refund_retrieve(state, _runtime(session_factory))
    assert update["evidence_strong"] is False
    assert update["refusal_source"] == "retrieval_low_conf"
    assert update["policy_refused"] is True


@pytest.mark.asyncio
async def test_log_turn_exposes_ch06_fields_on_trace():
    runtime = SimpleNamespace(context=SimpleNamespace(intent_mode="accuracy"))
    state = {
        "messages": [HumanMessage(content="订单1001能退吗")],
        "trace": {"t0": 0.0},
        "resolved_query": "订单1001能退吗",
        "coref_source": "model",
        "intent_confidence": 0.9,
        "order_no": "1001",
        "order_error": None,
        "policy_queries": ["q1"],
        "expand_degraded": False,
        "policy_refused": False,
        "refund_offered": True,
        "flow_steps": ["fetch_order"],
        "suggested_actions": [],
    }
    update = await nodes.log_turn(state, runtime)
    trace = update["trace"]
    assert trace["resolved_query"] == "订单1001能退吗"
    assert trace["coref_source"] == "model"
    assert trace["intent_confidence"] == 0.9
    assert trace["intent_mode"] == "accuracy"
    assert trace["order_no"] == "1001"
    assert trace["order_error"] is None
    assert trace["policy_queries"] == ["q1"]
    assert trace["expand_degraded"] is False
    assert trace["policy_refused"] is False
    assert trace["refund_offered"] is True
    assert trace["flow_steps"] == ["fetch_order"]
