"""图上的纯路由函数。不写 state，不碰 IO。"""

INTENTS = (
    "logistics", "order", "product_consult", "refund_return",
    "after_sale", "complaint", "chitchat",
)

ROUTE_BY_INTENT = {
    "logistics": "business",
    "order": "business",
    "product_consult": "knowledge",
    "refund_return": "knowledge",
    "after_sale": "business",
    "complaint": "complaint",
    "chitchat": "chitchat",
}


def route_by_intent(state) -> str:
    return state["route"]


def gate_router(state) -> str:
    if state.get("retrieval_error"):
        return "fallback"
    return "agent" if state.get("evidence_strong") else "fallback"


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
