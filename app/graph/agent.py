"""图级 ReAct 的两步：推理（agent_llm）与行动（agent_tools）。

`create_ticket` 在行动步被拦截成「前端按钮提议」：不写库、不改变会话状态，
只返回一条合成 ToolMessage 让模型收敛。
"""

import logging

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from app.prompts.customer_service import SYSTEM_PROMPT
from app.prompts.orchestrator import (
    AGENT_SYSTEM_ADDENDUM_BUSINESS,
    AGENT_SYSTEM_ADDENDUM_KNOWLEDGE,
)
from app.services.context_service import ContextStore
from app.tools.executor import execute_tool
from app.tools.registry import build_tools, tool_timeout

logger = logging.getLogger("knowweave.turn")


def build_agent_system_prompt(route: str, evidence: str | None) -> str:
    if route == "knowledge":
        addendum = AGENT_SYSTEM_ADDENDUM_KNOWLEDGE
        block = f"\n\n【本轮强制检索证据】\n{evidence or '（无）'}"
    else:
        addendum = AGENT_SYSTEM_ADDENDUM_BUSINESS
        block = ""
    return f"{SYSTEM_PROMPT}\n\n{addendum}{block}"


def _usage_tokens(ai: AIMessage) -> int:
    usage = getattr(ai, "usage_metadata", None) or {}
    return int(usage.get("total_tokens") or 0)


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content or "")


async def agent_llm(state, runtime):
    context = runtime.context
    route = state.get("route") or "business"
    tools = build_tools(
        context.session_factory,
        state["conversation_id"],
        context.turn,
        include_faq=(route == "business"),
    )
    system = SystemMessage(
        content=build_agent_system_prompt(route, state.get("evidence"))
    )
    history = list(state.get("messages") or [])
    trimmed = ContextStore(context.token_budget).trim(system, history)
    ai = await context.model.bind_tools(tools).ainvoke([system] + trimmed)
    update = {
        "messages": [ai],
        "steps": int(state.get("steps", 0)) + 1,
        "tokens_used": int(state.get("tokens_used", 0)) + _usage_tokens(ai),
        # 每轮重置：有工具调用时这一轮还没有最终答复。
        "answer": None,
    }
    if not ai.tool_calls:
        update["answer"] = _text_of(ai.content)
        update["stop_reason"] = "no_tool_calls"
    return update


def _intercept_create_ticket(tool_call, tools_by_name, actions):
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


async def agent_tools(state, runtime):
    context = runtime.context
    route = state.get("route") or "business"
    tools = build_tools(
        context.session_factory,
        state["conversation_id"],
        context.turn,
        include_faq=(route == "business"),
    )
    by_name = {tool.name: tool for tool in tools}
    actions = list(state.get("suggested_actions") or [])
    last = state["messages"][-1]
    tool_messages = []

    for call in getattr(last, "tool_calls", None) or []:
        if call.get("name") == "create_ticket":
            tool_messages.append(_intercept_create_ticket(call, by_name, actions))
            continue
        tool_messages.append(
            await execute_tool(
                call,
                by_name,
                timeout=tool_timeout(call["name"], context.tool_timeout),
                retries=context.tool_retries,
            )
        )

    citations = list(state.get("citations") or [])
    for citation in context.turn.citations:
        if citation not in citations:
            citations.append(citation)

    return {
        "messages": tool_messages,
        "suggested_actions": actions,
        "citations": citations,
    }
