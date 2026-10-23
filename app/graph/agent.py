"""图级 ReAct 的两步：推理（agent_llm）与行动（agent_tools）。

ch08：工具调用改走动态工具引擎。agent_llm 每次调用取一轮工具快照并绑定 JSON Schema；
agent_tools 每次节点执行只处理 pending_tool_calls 里游标指向的那一个调用，执行完推进
游标，由 make_tool_loop_router 决定回到 agent_tools（还有未执行调用）还是 agent_llm。
create_ticket 仍按旧策略在 agent 节点拦截成「前端按钮提议」，Task 9 前不写库。
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
from app.services.token_counter import TokenCounter, text_of
from app.tools.builtin import build_tool_snapshot, register_builtins
from app.tools.contracts import ExecutionContext
from app.tools.executor import ToolEngine
from app.tools.policy import PolicyStore
from app.tools.registry import ToolRegistry, tool_timeout
from app.tools.validation import argument_errors

logger = logging.getLogger("knowweave.turn")
context_logger = logging.getLogger("knowweave.context")


def _usage_tokens(ai: AIMessage) -> int:
    usage = getattr(ai, "usage_metadata", None) or {}
    return int(usage.get("total_tokens") or 0)


def _schema_of(definition) -> dict:
    """把 ToolDefinition 转成 bind_tools 可接受的 OpenAI function 风格 dict。"""
    return {
        "name": definition.name,
        "description": definition.description,
        "parameters": definition.parameters,
    }


def _resolve_registry(context) -> ToolRegistry:
    """context.registry 为空时回退到「仅内置工具」注册表，保证直连单测可用。"""
    registry = getattr(context, "registry", None)
    if registry is None:
        registry = ToolRegistry()
        register_builtins(registry)
    return registry


def _resolve_engine(context) -> ToolEngine:
    """context.engine 为空时构造一个仅放行内置工具的引擎（直连单测兜底）。"""
    engine = getattr(context, "engine", None)
    if engine is not None:
        return engine
    from app.tools.audit import write_audit

    session_factory = getattr(context, "session_factory", None)
    default = getattr(context, "tool_timeout", 5.0)
    policy = PolicyStore("")  # 空路径加载失败 → 快照只放行内置工具
    return ToolEngine(
        policy,
        lambda record: write_audit(session_factory, record),
        timeout=lambda name: tool_timeout(name, default),
        retries=getattr(context, "tool_retries", 1),
    )


async def agent_llm(state, runtime):
    context = runtime.context
    route = state.get("route") or "business"
    intent = state.get("intent")
    registry = _resolve_registry(context)
    provider = getattr(context, "provider", None)
    definitions = await build_tool_snapshot(registry, provider)
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

    raw_schemas = [_schema_of(d) for d in definitions.values()]
    schemas = [convert_to_openai_tool(schema) for schema in raw_schemas]
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
        "tools": list(definitions),
        "tool_schemas": schemas,
        "budget": getattr(getattr(context, "budget", None), "as_dict", lambda: {})() if getattr(context, "budget", None) else {},
        "anchors": {
            "summary_upto": meta.get("summary_upto"),
            "layer1_from": meta.get("layer1_from"),
        },
        "route": route,
        "intent": intent,
    }, ensure_ascii=False))

    ai = await context.model.bind_tools(raw_schemas).ainvoke(sent)
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
    }
    if ai.tool_calls:
        # 有工具调用：进入工具循环，重置队列与游标；本轮还没有最终答复。
        update.update({
            "answer": None,
            "pending_tool_calls": list(ai.tool_calls),
            "tool_cursor": 0,
            "tool_tokens_used": 0,
        })
    else:
        update.update({
            "answer": text_of(ai.content),
            "stop_reason": "no_tool_calls",
            "pending_tool_calls": [],
            "tool_cursor": 0,
            "tool_tokens_used": 0,
        })
    return update


def _format_argument_errors(errors: list[dict]) -> str:
    parts = []
    for e in errors:
        path = "/".join(str(p) for p in e["path"]) or "(root)"
        message = e["message"]
        if "allowed_values" in e:
            message += f"（允许值: {', '.join(map(str, e['allowed_values']))}）"
        parts.append(f"{path}: {message}")
    return "参数校验失败: " + "; ".join(parts)


def _intercept_create_ticket(call, definitions, *, intent=None) -> ToolMessage:
    """create_ticket 保持旧拦截策略：不写库，只产出前端按钮提议（Task 9 前）。

    参数改用 definition.parameters 做严格 JSON Schema 校验（不再是旧 Pydantic args_schema）。
    """
    call_id = call.get("id", "")
    name = "create_ticket"
    definition = definitions.get("create_ticket")
    if definition is None:
        return ToolMessage(content="未知工具: create_ticket", tool_call_id=call_id, name=name)

    args = call.get("args") or {}
    errors = argument_errors(definition.parameters, args)
    if errors:
        return ToolMessage(
            content=_format_argument_errors(errors),
            tool_call_id=call_id,
            name=name,
        )

    description = args.get("description", "")
    ticket_type = args.get("ticket_type", "咨询")
    produced = []
    if intent == "after_sale" and ticket_type == "售后":
        produced.append({"type": "transfer_human", "label": "转人工"})
    produced.append({
        "type": "create_ticket",
        "label": "建工单",
        "draft": {"description": description, "ticket_type": ticket_type},
    })
    return ToolMessage(
        content="工单选项已呈现给用户，请简要说明并停止，不要再次调用 create_ticket。",
        tool_call_id=call_id,
        name=name,
        artifact={"category": "success", "actions": produced},
    )


async def agent_tools(state, runtime):
    context = runtime.context
    registry = _resolve_registry(context)
    provider = getattr(context, "provider", None)
    definitions = await build_tool_snapshot(registry, provider)
    engine = _resolve_engine(context)

    pending = list(state.get("pending_tool_calls") or [])
    cursor = int(state.get("tool_cursor") or 0)
    if cursor >= len(pending):
        # 防御性：正常由 tool_loop_router 拦住，不会执行到这里。
        return {
            "messages": [],
            "tool_cursor": cursor,
            "suggested_actions": list(state.get("suggested_actions") or []),
            "citations": list(state.get("citations") or []),
            "refund_offered": bool(state.get("refund_offered")),
        }

    call = pending[cursor]
    name = call.get("name", "")
    actions = list(state.get("suggested_actions") or [])
    counter = getattr(context, "counter", None) or TokenCounter()
    budget = int(getattr(context, "tool_result_max_tokens", 1200))
    tool_tokens_used = int(state.get("tool_tokens_used") or 0)

    # 每次执行前按剩余调用数分配本轮预算（含当前这一条）。
    remaining_calls = len(pending) - cursor
    share = max(0, budget - tool_tokens_used) // remaining_calls if remaining_calls else 0

    if name == "create_ticket":
        message = _intercept_create_ticket(call, definitions, intent=state.get("intent"))
    else:
        exec_context = ExecutionContext(
            conversation_id=state.get("conversation_id"),
            tool_call_id=call.get("id", ""),
            session_factory=context.session_factory,
            turn=context.turn,
            state=state,
        )
        message = await engine.execute(call, exec_context, definitions)

    # 单条结果截断（与旧批量实现共用同一预算口径）。
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
            "tool": name, "cap": share,
        }, ensure_ascii=False))

    tool_tokens_used += counter.count_message(message)

    # 从 ToolMessage.artifact 收集前端动作（去重），替代旧解析。
    artifact = getattr(message, "artifact", None) or {}
    new_actions = artifact.get("actions", [])
    for action in new_actions:
        if action not in actions:
            actions.append(action)

    citations = list(state.get("citations") or [])
    for citation in getattr(context.turn, "citations", None) or []:
        if citation not in citations:
            citations.append(citation)

    refund_offered = bool(state.get("refund_offered"))
    if any(isinstance(a, dict) and a.get("type") == "refund_form" for a in new_actions):
        refund_offered = True

    return {
        "messages": [message],
        "tool_cursor": cursor + 1,
        "tool_tokens_used": tool_tokens_used,
        "suggested_actions": actions,
        "citations": citations,
        "refund_offered": refund_offered,
    }
