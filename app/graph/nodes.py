"""图节点。"""

import asyncio
import json
import logging
import time

from langchain_core.messages import AIMessage, HumanMessage

from app.db.repository import LowConfidenceRepository
from app.graph.routing import ROUTE_BY_INTENT
from app.prompts.orchestrator import (
    AGENT_STOPPED_REPLY,
    CHITCHAT_REPLY,
    COMPLAINT_REPLY,
    KNOWLEDGE_UNAVAILABLE_REPLY,
    CorefOutput,
    coref_messages,
    intent_messages,
)
from app.rag.pipeline import run_pipeline
from app.rag.refusal import SOURCE_RETRIEVAL, pool_record, refusal_text

logger = logging.getLogger("knowweave.turn")


def _last_user_text(messages) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


async def classify_intent(state, runtime):
    question = _last_user_text(state.get("messages", []))
    try:
        result = await runtime.context.intent_model.ainvoke(intent_messages(question))
        intent = getattr(result, "intent", None)
        if intent not in ROUTE_BY_INTENT:
            raise ValueError(f"未知意图: {intent!r}")
        source = "model"
    except Exception:
        logger.warning("意图分类失败，兜底 after_sale", exc_info=True)
        intent = "after_sale"
        source = "fallback"
    return {
        "intent": intent,
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
    history = messages[:-1][-8:] if messages and isinstance(messages[-1], HumanMessage) else messages[-8:]
    model = getattr(runtime.context, "coref_model", None)
    if model is None:
        return {"resolved_query": current, "coref_source": "passthrough"}
    try:
        result = await model.ainvoke(coref_messages(history, current))
        resolved = (getattr(result, "resolved_query", "") or "").strip()
        if not resolved:
            raise ValueError("resolved_query 为空")
    except Exception:
        logger.warning("指代消解失败，原话透传", exc_info=True)
        return {"resolved_query": current, "coref_source": "fallback"}
    source = "passthrough" if _norm_question(resolved) == _norm_question(current) else "model"
    return {"resolved_query": resolved, "coref_source": source}


async def forced_rag(state, runtime):
    question = _last_user_text(state.get("messages", []))
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
        "actions": [a.get("type") for a in state.get("suggested_actions", [])],
        "tools": trace["tools"],
        "latency_ms": trace["latency_ms"],
    }
    logger.info(json.dumps(payload, ensure_ascii=False))
    return {"trace": trace}
