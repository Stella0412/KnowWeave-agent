"""图上的纯路由函数。不写 state，不碰 IO。"""

INTENTS = (
    "logistics", "order", "product_consult", "refund_return",
    "after_sale", "complaint", "chitchat", "other",
)

ROUTE_BY_INTENT = {
    "logistics": "business",
    "order": "business",
    "product_consult": "knowledge",
    "refund_return": "refund_flow",
    "after_sale": "refund_flow",
    "complaint": "complaint",
    "chitchat": "chitchat",
    "other": "clarify",
}


def route_by_intent(state) -> str:
    return state["route"]


def gate_router(state) -> str:
    if state.get("retrieval_error"):
        return "fallback"
    return "agent" if state.get("evidence_strong") else "fallback"


def refund_order_router(state) -> str:
    return "fallback" if state.get("order_error") else "expand"


def make_should_continue(max_steps: int):
    def should_continue(state) -> str:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        tool_calls = getattr(last, "tool_calls", None) if last is not None else None
        if not tool_calls:
            return "log"
        if int(state.get("steps", 0)) >= max_steps:
            return "fallback"
        return "tools"

    return should_continue


def make_tool_loop_router():
    """agent_tools 之后的循环路由：游标未走完继续执行下一个调用，否则回到 agent_llm。"""

    def tool_loop_router(state) -> str:
        cursor = int(state.get("tool_cursor") or 0)
        pending = state.get("pending_tool_calls") or []
        return "tools" if cursor < len(pending) else "agent_llm"

    return tool_loop_router
