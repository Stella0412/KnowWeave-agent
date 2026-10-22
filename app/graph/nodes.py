"""图节点。"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from app.db.repository import LowConfidenceRepository, OrderRepository
from app.graph.routing import ROUTE_BY_INTENT
from app.prompts.orchestrator import (
    AGENT_STOPPED_REPLY,
    CHITCHAT_REPLY,
    COMPLAINT_REPLY,
    KNOWLEDGE_UNAVAILABLE_REPLY,
    NO_ORDERS_REPLY,
    ORDER_NOT_FOUND_REPLY,
    CorefOutput,
    coref_messages,
    expansion_messages,
    intent_messages,
)
from app.rag.pipeline import run_expanded_pipeline, run_pipeline
from app.rag.refusal import SOURCE_RETRIEVAL, pool_record, refusal_text

logger = logging.getLogger("knowweave.turn")


def _last_user_text(messages) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _current_user_msg_id(messages) -> int | None:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            raw = getattr(message, "id", None)
            if isinstance(raw, str) and raw.startswith("db:"):
                try:
                    return int(raw[3:])
                except ValueError:
                    return None
    return None


async def prepare_context(state, runtime):
    manager = getattr(runtime.context, "context_manager", None)
    if manager is None:
        return {"context_meta": {}}
    current_id = state.get("turn_user_msg_id") or _current_user_msg_id(
        state.get("messages", [])
    )
    if current_id is None:
        return {"context_meta": {}}
    meta = await manager.prepare(
        state["conversation_id"], state.get("messages", []), current_id
    )
    return {"context_meta": meta}


async def classify_intent(state, runtime):
    question = state.get("resolved_query") or _last_user_text(state.get("messages", []))
    meta = state.get("context_meta") or {}
    manager = getattr(runtime.context, "context_manager", None)
    if manager is not None and meta:
        history = manager.shared_history(state.get("messages", []), meta)
    else:
        messages = list(state.get("messages") or [])
        history = messages[:-1][-6:] if messages and isinstance(messages[-1], HumanMessage) else messages[-6:]
    try:
        result = await runtime.context.intent_model.ainvoke(
            intent_messages(question, history, meta.get("summary_text", ""))
        )
        intent = getattr(result, "intent", None)
        confidence = float(getattr(result, "confidence", None))
        if intent not in ROUTE_BY_INTENT:
            raise ValueError(f"未知意图: {intent!r}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"置信度越界: {confidence!r}")
        source = "model"
    except Exception:
        logger.warning("意图分类失败，兜底 other", exc_info=True)
        intent = "other"
        confidence = None
        source = "fallback"
    return {
        "intent": intent,
        "intent_confidence": confidence,
        "route": ROUTE_BY_INTENT[intent],
        "intent_source": source,
    }


def _norm_question(text: str) -> str:
    return (text or "").strip().rstrip("。！？!?")


async def coref(state, runtime):
    messages = list(state.get("messages") or [])
    current = _last_user_text(messages)
    if not current:
        return {"resolved_query": "", "coref_source": "passthrough"}
    meta = state.get("context_meta") or {}
    manager = getattr(runtime.context, "context_manager", None)
    if manager is not None and meta:
        history = manager.shared_history(messages, meta)
    else:
        history = messages[:-1][-8:] if messages and isinstance(messages[-1], HumanMessage) else messages[-8:]
    model = getattr(runtime.context, "coref_model", None)
    if model is None:
        return {"resolved_query": current, "coref_source": "passthrough"}
    try:
        result = await model.ainvoke(
            coref_messages(history, current, meta.get("summary_text", ""))
        )
        resolved = (getattr(result, "resolved_query", "") or "").strip()
        if not resolved:
            raise ValueError("resolved_query 为空")
    except Exception:
        logger.warning("指代消解失败，原话透传", exc_info=True)
        return {"resolved_query": current, "coref_source": "fallback"}
    source = "passthrough" if _norm_question(resolved) == _norm_question(current) else "model"
    return {"resolved_query": resolved, "coref_source": source}


def _unique_queries(raw, fallback: str) -> tuple[list[str], bool]:
    out: list[str] = []
    for item in raw or []:
        q = str(item or "").strip()
        if q and q not in out:
            out.append(q)
    if not out:
        # 空输出按失败处理：退化为单查询，不拿空串或重复串去召回。
        return [fallback], True
    degraded = len(out) < 3
    suffixes = ("政策条件", "时限与起算点", "例外情形")
    while len(out) < 3:
        out.append(f"{fallback}（{suffixes[len(out)]}）")
    return out[:3], degraded


async def refund_expand(state, runtime):
    question = state.get("resolved_query") or _last_user_text(state.get("messages", []))
    trace = dict(state.get("trace") or {})
    model = getattr(runtime.context, "expansion_model", None)
    degraded = False
    if model is None:
        queries = [question]
        degraded = True
    else:
        try:
            result = await model.ainvoke(
                expansion_messages(question, state.get("intent") or "", state.get("order_data"))
            )
            queries, degraded = _unique_queries(getattr(result, "queries", []), question)
        except Exception:
            logger.warning("查询扩写失败，退化为单查询", exc_info=True)
            queries = [question]
            degraded = True
    trace["expansion_degraded"] = degraded
    steps = list(state.get("flow_steps") or [])
    if "expand" not in steps:
        steps.append("expand")
    return {
        "policy_queries": queries,
        "expand_degraded": degraded,
        "flow_steps": steps,
        "trace": trace,
    }


def _iso(value) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _order_picker_item(order) -> dict:
    return {
        "order_no": order.order_no,
        "product_name": order.product_name,
        "status": order.status,
        "amount": order.amount,
        "signed_at": _iso(order.signed_at),
    }


def _order_data(order) -> dict:
    return {
        **_order_picker_item(order),
        "product_type": order.product_type,
        "tracking_no": order.tracking_no,
        "paid_at": _iso(order.paid_at),
    }


def _order_numbers(text: str, allowed: set[str]) -> list[str]:
    found = []
    for match in re.findall(r"(?<!\d)(\d{4,8})(?!\d)", text or ""):
        if match in allowed and match not in found:
            found.append(match)
    return found


async def refund_fetch_order(state, runtime):
    trace = dict(state.get("trace") or {})
    steps = list(state.get("flow_steps") or [])
    async with runtime.context.session_factory() as session:
        orders = await OrderRepository(session).list_for_user(
            state.get("user_id") or "anonymous"
        )
    if not orders:
        trace["order_error"] = "empty"
        return {
            "order_error": "empty",
            "fallback_reason": "order_unavailable",
            "stop_reason": "order_unavailable",
            "flow_steps": steps,
            "trace": trace,
        }

    allowed = {order.order_no for order in orders}
    texts = []
    for candidate in (
        state.get("resolved_query"),
        _last_user_text(state.get("messages", [])),
    ):
        if candidate and candidate not in texts:
            texts.append(candidate)
    order_no = next(
        (no for text in texts for no in _order_numbers(text, allowed)), None
    )
    if order_no is None:
        choice = interrupt(
            {
                "kind": "select_order",
                "prompt": (
                    "请选择要退款的订单"
                    if state.get("intent") == "refund_return"
                    else "请选择要处理的订单"
                ),
                "orders": [_order_picker_item(order) for order in orders],
            }
        )
        order_no = str((choice or {}).get("order_no") or "").strip()

    async with runtime.context.session_factory() as session:
        order = await OrderRepository(session).get(order_no)
    if order is None:
        trace["order_error"] = "not_found"
        return {
            "order_error": "not_found",
            "fallback_reason": "order_unavailable",
            "stop_reason": "order_unavailable",
            "flow_steps": steps,
            "trace": trace,
        }

    steps.append("fetch_order")
    return {
        "order_no": order.order_no,
        "order_data": _order_data(order),
        "order_error": None,
        "flow_steps": steps,
        "trace": trace,
    }


async def refund_retrieve(state, runtime):
    trace = dict(state.get("trace") or {})
    steps = list(state.get("flow_steps") or [])
    query = state.get("resolved_query") or _last_user_text(state.get("messages", []))
    queries = list(state.get("policy_queries") or [query])
    trace["policy_queries"] = queries
    try:
        result = await asyncio.to_thread(
            run_expanded_pipeline,
            queries,
            query,
            source_files=runtime.context.turn.source_files or None,
        )
    except Exception as exc:
        logger.warning("政策检索失败", exc_info=True)
        trace["retrieval_error"] = str(exc)
        return {
            "evidence": "",
            "citations": [],
            "evidence_strong": False,
            "retrieval_error": str(exc),
            "refusal_source": None,
            "refusal_reason": None,
            "policy_refused": None,
            "flow_steps": steps,
            "trace": trace,
        }

    strong = (not result.refused) and bool(result.evidence_text)
    trace["policy_refused"] = result.refused
    trace["rerank_score"] = result.rerank_score
    steps.append("retrieve")
    return {
        "evidence": result.evidence_text,
        "citations": result.citations,
        "evidence_strong": strong,
        "retrieval_error": None,
        "refusal_source": result.refusal_source,
        "refusal_reason": result.refusal_reason,
        "policy_refused": result.refused,
        "flow_steps": steps,
        "trace": trace,
    }


async def forced_rag(state, runtime):
    question = state.get("resolved_query") or _last_user_text(state.get("messages", []))
    trace = dict(state.get("trace") or {})
    trace["forced_rag"] = True
    try:
        result = await asyncio.to_thread(
            run_pipeline,
            question,
            source_files=runtime.context.turn.source_files or None,
        )
    except Exception as exc:
        logger.warning("知识库强制检索失败", exc_info=True)
        trace["retrieval_error"] = str(exc)
        return {
            "evidence": "",
            "citations": [],
            "evidence_strong": False,
            "retrieval_error": str(exc),
            "refusal_source": None,
            "refusal_reason": None,
            "trace": trace,
        }

    strong = (not result.refused) and bool(result.evidence_text)
    trace["retrieval_refused"] = result.refused
    trace["rerank_score"] = result.rerank_score
    return {
        "evidence": result.evidence_text,
        "citations": result.citations,
        "evidence_strong": strong,
        "retrieval_error": None,
        "refusal_source": result.refusal_source,
        "refusal_reason": result.refusal_reason,
        "trace": trace,
    }


async def confidence_check(state, runtime):
    trace = dict(state.get("trace") or {})
    if state.get("retrieval_error"):
        trace["gate"] = "retrieval_unavailable"
        return {
            "fallback_reason": "retrieval_unavailable",
            "stop_reason": "retrieval_unavailable",
            "trace": trace,
        }
    if not state.get("evidence_strong"):
        trace["gate"] = "evidence_weak"
        return {
            "fallback_reason": "evidence_weak",
            "stop_reason": "gate_refused",
            "trace": trace,
        }
    trace["gate"] = "passed"
    return {"trace": trace}


async def _record_low_confidence(
    session_factory, conversation_id, question, source, reason
):
    if session_factory is None:
        return
    record = pool_record(question, source or SOURCE_RETRIEVAL, reason)
    try:
        async with session_factory() as session:
            await LowConfidenceRepository(session).add(conversation_id, **record)
            await session.commit()
    except Exception:
        logger.warning("低置信度问题落池失败（不影响本轮回答）", exc_info=True)


async def fallback_reply(state, runtime):
    question = _last_user_text(state.get("messages", []))
    reason = state.get("fallback_reason") or "agent_stopped"
    trace = dict(state.get("trace") or {})
    trace["fallback_reason"] = reason

    if reason == "retrieval_unavailable":
        text = KNOWLEDGE_UNAVAILABLE_REPLY
        stop_reason = state.get("stop_reason") or "retrieval_unavailable"
    elif reason == "agent_stopped":
        text = AGENT_STOPPED_REPLY
        stop_reason = "max_steps"
    elif reason == "order_unavailable":
        text = NO_ORDERS_REPLY if state.get("order_error") == "empty" else ORDER_NOT_FOUND_REPLY
        stop_reason = "order_unavailable"
    else:
        text = refusal_text(question)
        stop_reason = state.get("stop_reason") or "gate_refused"
        await _record_low_confidence(
            runtime.context.session_factory,
            state.get("conversation_id"),
            question,
            state.get("refusal_source"),
            state.get("refusal_reason"),
        )

    return {
        "messages": [AIMessage(content=text)],
        "answer": text,
        "stop_reason": stop_reason,
        "trace": trace,
    }


async def complaint_reply(state, runtime):
    question = _last_user_text(state.get("messages", [])) or "用户投诉"
    actions = [
        {"type": "transfer_human", "label": "转人工"},
        {
            "type": "create_ticket",
            "label": "建工单",
            "draft": {"description": question, "ticket_type": "投诉"},
        },
    ]
    trace = dict(state.get("trace") or {})
    trace["fixed_reply"] = "complaint"
    return {
        "messages": [AIMessage(content=COMPLAINT_REPLY)],
        "answer": COMPLAINT_REPLY,
        "suggested_actions": actions,
        "stop_reason": "fixed_reply",
        "trace": trace,
    }


async def chitchat_reply(state, runtime):
    trace = dict(state.get("trace") or {})
    trace["fixed_reply"] = "chitchat"
    return {
        "messages": [AIMessage(content=CHITCHAT_REPLY)],
        "answer": CHITCHAT_REPLY,
        "suggested_actions": [],
        "stop_reason": "fixed_reply",
        "trace": trace,
    }


def _tool_names(messages) -> list[str]:
    names = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name")
            if name:
                names.append(name)
    return names


async def log_turn(state, runtime):
    trace = dict(state.get("trace") or {})
    t0 = trace.get("t0")
    # messages 里带着 checkpointer 的历史，只统计本轮新增的消息，
    # 否则第 N 轮的 trace 会把前几轮的工具调用也重复算进去。
    offset = int(state.get("turn_msg_offset") or 0)
    trace["tools"] = _tool_names(state.get("messages", [])[offset:])
    trace["latency_ms"] = int((time.perf_counter() - t0) * 1000) if t0 else 0
    payload = {
        "conversation_id": state.get("conversation_id"),
        "intent": state.get("intent"),
        "intent_source": state.get("intent_source"),
        "route": state.get("route"),
        "forced_rag": bool(trace.get("forced_rag")),
        "retrieval_refused": trace.get("retrieval_refused"),
        "rerank_score": trace.get("rerank_score"),
        "evidence_strong": state.get("evidence_strong"),
        "steps": state.get("steps", 0),
        "tokens_used": state.get("tokens_used", 0),
        "stop_reason": state.get("stop_reason"),
        "fallback_reason": state.get("fallback_reason"),
        "resolved_query": state.get("resolved_query"),
        "coref_source": state.get("coref_source"),
        "intent_confidence": state.get("intent_confidence"),
        "intent_mode": getattr(runtime.context, "intent_mode", "accuracy"),
        "order_no": state.get("order_no"),
        "order_error": state.get("order_error"),
        "policy_queries": state.get("policy_queries") or [],
        "expand_degraded": state.get("expand_degraded"),
        "policy_refused": state.get("policy_refused"),
        "refund_offered": state.get("refund_offered"),
        "flow_steps": state.get("flow_steps") or [],
        "actions": [a.get("type") for a in state.get("suggested_actions", [])],
        "tools": trace["tools"],
        "latency_ms": trace["latency_ms"],
    }
    logger.info(json.dumps(payload, ensure_ascii=False))
    # ch06 验收口径：/chat/agent 与手动验收读 trace.resolved_query/order_no/flow_steps/policy_queries。
    # 这些字段此前只进 JSON 日志 payload，不随 trace 返回；这里原样并入 trace，日志不变。
    trace.update(
        {
            "resolved_query": state.get("resolved_query"),
            "coref_source": state.get("coref_source"),
            "intent_confidence": state.get("intent_confidence"),
            "intent_mode": getattr(runtime.context, "intent_mode", "accuracy"),
            "order_no": state.get("order_no"),
            "order_error": state.get("order_error"),
            "policy_queries": state.get("policy_queries") or [],
            "expand_degraded": state.get("expand_degraded"),
            "policy_refused": state.get("policy_refused"),
            "refund_offered": state.get("refund_offered"),
            "flow_steps": state.get("flow_steps") or [],
        }
    )
    return {"trace": trace}
