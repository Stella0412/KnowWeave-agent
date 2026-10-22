"""图级 ReAct 的两步：推理（agent_llm）与行动（agent_tools）。

`create_ticket` 在行动步被拦截成「前端按钮提议」：不写库、不改变会话状态，
只返回一条合成 ToolMessage 让模型收敛。
"""

import json
import logging

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.prompts.context import build_dynamic_block
from app.prompts.customer_service import SYSTEM_PROMPT
from app.services.context_manager import serialize_message, split_turns
from app.services.refund_policy import check_refund_eligibility
from app.services.token_counter import TokenCounter, text_of
from app.tools.business import REFUND_REASON_CATEGORIES
from app.tools.executor import execute_tool
from app.tools.registry import allowed_tool_names, build_tools, tool_timeout

logger = logging.getLogger("knowweave.turn")
context_logger = logging.getLogger("knowweave.context")


def _usage_tokens(ai: AIMessage) -> int:
    usage = getattr(ai, "usage_metadata", None) or {}
    return int(usage.get("total_tokens") or 0)


async def agent_llm(state, runtime):
    context = runtime.context
    route = state.get("route") or "business"
    intent = state.get("intent")
    tools = build_tools(context.session_factory, state["conversation_id"], context.turn)
    messages = list(state.get("messages") or [])
    meta = state.get("context_meta") or {}
    manager = getattr(context, "context_manager", None)
    if manager is not None and meta:
        prefix, current, react = manager.model_history(messages, meta)
    else:
        turns = split_turns(messages)
        current_turn = turns[-1] if turns else None
        prefix = [m for turn in turns[:-1] for m in turn.messages]
        current = current_turn.messages[0] if current_turn and current_turn.messages else None
        react = current_turn.messages[1:] if current_turn else []
        if not meta:
            context_logger.warning(json.dumps({
                "event": "context_meta_missing",
                "conversation_id": state.get("conversation_id"),
            }, ensure_ascii=False))

    # coref 已经把「那大概什么时候能到」补全成本轮真正的问题；发给模型的本轮用户消息
    # 必须是消解后的完整问句，状态/落库里的用户原话保持不变。
    resolved = (state.get("resolved_query") or "").strip()
    current_text = resolved or text_of(getattr(current, "content", ""))
    if route == "refund_flow" and state.get("order_no"):
        current_text += (
            f"\n【本轮已确认订单】{state['order_no']}。"
            "请仅处理这张订单，以系统提供的本轮订单数据为准。"
        )
    counter = getattr(context, "counter", None) or TokenCounter()
    context_settings = getattr(context, "settings", None)
    dynamic = build_dynamic_block(
        route=route,
        intent=intent,
        summary=meta.get("summary_text", ""),
        evidence=state.get("evidence"),
        order_data=state.get("order_data"),
        counter=counter,
        evidence_budget_tokens=(
            context_settings.rerank_top_n * context_settings.evidence_tokens_per_chunk
            if context_settings is not None
            else 10 * 200
        ),
        order_max_tokens=(
            context_settings.order_json_max_tokens
            if context_settings is not None
            else 600
        ),
    )
    if dynamic:
        current_text += "\n\n" + dynamic

    sent = [SystemMessage(content=SYSTEM_PROMPT)] + prefix
    if current is not None:
        sent.append(HumanMessage(content=current_text, id=getattr(current, "id", None)))
    sent.extend(react)

    schemas = [convert_to_openai_tool(tool) for tool in tools]
    schema_text = json.dumps(schemas, ensure_ascii=False, sort_keys=True)
    estimated_messages_tokens = counter.count_messages(sent)
    estimated_tool_tokens = counter.count_text(schema_text)
    estimated_tokens = estimated_messages_tokens + estimated_tool_tokens
    context_logger.info(json.dumps({
        "event": "model_ctx",
        "conversation_id": state.get("conversation_id"),
        "step": int(state.get("steps", 0)) + 1,
        "system": SYSTEM_PROMPT,
        "summary": meta.get("summary_text", ""),
        "summary_segments": meta.get("summary_segments", []),
        "messages": [serialize_message(m) for m in sent],
        "window_count": len(sent),
        "estimated_tokens": estimated_messages_tokens,
        "estimated_tool_tokens": estimated_tool_tokens,
        "estimated_total_tokens": estimated_tokens,
        "tools": [tool.name for tool in tools],
        "tool_schemas": schemas,
        "budget": getattr(getattr(context, "budget", None), "as_dict", lambda: {})() if getattr(context, "budget", None) else {},
        "anchors": {
            "summary_upto": meta.get("summary_upto"),
            "layer1_from": meta.get("layer1_from"),
        },
        "route": route,
        "intent": intent,
    }, ensure_ascii=False))

    ai = await context.model.bind_tools(tools).ainvoke(sent)
    actual_tokens = _usage_tokens(ai)
    if actual_tokens:
        context_logger.info(json.dumps({
            "event": "model_usage",
            "conversation_id": state.get("conversation_id"),
            "step": int(state.get("steps", 0)) + 1,
            "estimated_messages_tokens": estimated_messages_tokens,
            "estimated_tool_tokens": estimated_tool_tokens,
            "estimated_tokens": estimated_tokens,
            "actual_tokens": actual_tokens,
            "ratio": round(actual_tokens / max(1, estimated_tokens), 3),
        }, ensure_ascii=False))
    update = {
        "messages": [ai],
        "steps": int(state.get("steps", 0)) + 1,
        "tokens_used": int(state.get("tokens_used", 0)) + _usage_tokens(ai),
        # 每轮重置：有工具调用时这一轮还没有最终答复。
        "answer": None,
    }
    if not ai.tool_calls:
        update["answer"] = text_of(ai.content)
        update["stop_reason"] = "no_tool_calls"
    return update


def _intercept_create_ticket(tool_call, tools_by_name, actions, *, intent=None):
    call_id = tool_call.get("id", "")
    tool = tools_by_name.get("create_ticket")
    if tool is None:
        return ToolMessage(content="未知工具: create_ticket", tool_call_id=call_id, name="create_ticket")
    try:
        parsed = tool.args_schema.model_validate(tool_call.get("args") or {})
    except Exception as exc:
        return ToolMessage(
            content=f"工具参数校验失败: {exc}",
            tool_call_id=call_id,
            name="create_ticket",
        )

    description = getattr(parsed, "description", "")
    ticket_type = getattr(parsed, "ticket_type", "咨询")
    if intent == "after_sale" and ticket_type == "售后":
        transfer = {"type": "transfer_human", "label": "转人工"}
        if transfer not in actions:
            actions.append(transfer)
    action = {
        "type": "create_ticket",
        "label": "建工单",
        "draft": {"description": description, "ticket_type": ticket_type},
    }
    if action not in actions:
        actions.append(action)
    return ToolMessage(
        content="工单选项已呈现给用户，请简要说明并停止，不要再次调用 create_ticket。",
        tool_call_id=call_id,
        name="create_ticket",
    )


def _intercept_submit_refund(tool_call, tools_by_name, actions, state):
    call_id = tool_call.get("id", "")
    tool = tools_by_name.get("submit_refund")
    if tool is None:
        return ToolMessage(content="未知工具: submit_refund", tool_call_id=call_id), False
    try:
        parsed = tool.args_schema.model_validate(tool_call.get("args") or {})
    except Exception as exc:
        return ToolMessage(
            content=f"工具参数校验失败: {exc}", tool_call_id=call_id, name="submit_refund"
        ), False
    order_no = str(getattr(parsed, "order_no", "") or "").strip()
    if order_no != str(state.get("order_no") or ""):
        return ToolMessage(
            content="退款单订单号与本轮订单不一致，不能生成表单。",
            tool_call_id=call_id, name="submit_refund",
        ), False
    order = state.get("order_data") or {}
    # 模型判断可能出错（例如把定制类当成普通商品）；表单只对确定能无理由退货的订单生成。
    eligible, reason = check_refund_eligibility(order)
    if not eligible:
        return ToolMessage(
            content=f"该订单不能生成退款表单：{reason}。不要调用 submit_refund，请向用户说明原因。",
            tool_call_id=call_id, name="submit_refund",
        ), False
    action = {
        "type": "refund_form",
        "label": "提交退款单",
        "draft": {
            "order_no": order_no,
            "product_name": order.get("product_name", ""),
            "reason_options": list(REFUND_REASON_CATEGORIES),
        },
    }
    if action not in actions:
        actions.append(action)
    return ToolMessage(
        # 这条消息会随历史落库。「不要再次调用」若不限定本轮，下一轮用户再要求退款时
        # 模型会以为表单还在，既不调用工具、前端也就拿不到新的 refund_form 动作。
        content=(
            "本轮退款表单已呈现给用户，请简要说明并停止，本轮不要再次调用 submit_refund。"
            "用户下一轮再次要求退款时，表单不会保留，必须重新调用 submit_refund。"
        ),
        tool_call_id=call_id, name="submit_refund",
    ), True


async def agent_tools(state, runtime):
    context = runtime.context
    route = state.get("route") or "business"
    intent = state.get("intent")
    tools = build_tools(
        context.session_factory, state["conversation_id"], context.turn
    )
    by_name = {tool.name: tool for tool in tools}
    allowed = allowed_tool_names(route, intent)
    actions = list(state.get("suggested_actions") or [])
    last = state["messages"][-1]
    calls = list(getattr(last, "tool_calls", None) or [])
    counter = getattr(context, "counter", None) or TokenCounter()
    budget = int(getattr(context, "tool_result_max_tokens", 1200))
    results: list[ToolMessage | None] = [None] * len(calls)
    to_execute: list[int] = []
    special_tokens = 0
    refund_offered = bool(state.get("refund_offered"))

    for index, call in enumerate(calls):
        name = call.get("name", "")
        call_id = call.get("id", "")
        if name not in allowed:
            message = ToolMessage(
                content=f"本轮不允许调用 {name}", tool_call_id=call_id, name=name
            )
            results[index] = message
            special_tokens += counter.count_message(message)
            logger.info(json.dumps({
                "event": "tool_blocked", "conversation_id": state.get("conversation_id"),
                "route": route, "intent": intent, "tool": name,
            }, ensure_ascii=False))
            continue
        if name == "create_ticket":
            message = _intercept_create_ticket(call, by_name, actions, intent=intent)
        elif name == "submit_refund":
            message, offered = _intercept_submit_refund(call, by_name, actions, state)
            refund_offered = refund_offered or offered
        else:
            to_execute.append(index)
            continue
        results[index] = message
        special_tokens += counter.count_message(message)

    remaining = max(0, budget - special_tokens)
    share = remaining // len(to_execute) if to_execute else 0
    for index in to_execute:
        call = calls[index]
        message = await execute_tool(
            call,
            by_name,
            timeout=tool_timeout(call.get("name", ""), context.tool_timeout),
            retries=context.tool_retries,
        )
        if counter.count_message(message) > share:
            overhead = (
                4
                + counter.count_text(message.name or "")
                + counter.count_text(message.tool_call_id or "")
            )
            content = counter.truncate_text(
                text_of(message.content), max(0, share - overhead), "…（工具结果已截断）"
            )
            message = message.model_copy(update={"content": content})
            logger.info(json.dumps({
                "event": "tool_result_truncated",
                "conversation_id": state.get("conversation_id"),
                "tool": call.get("name", ""), "cap": share,
            }, ensure_ascii=False))
        results[index] = message

    citations = list(state.get("citations") or [])
    for citation in context.turn.citations:
        if citation not in citations:
            citations.append(citation)

    return {
        "messages": [message for message in results if message is not None],
        "suggested_actions": actions,
        "citations": citations,
        "refund_offered": refund_offered,
    }
