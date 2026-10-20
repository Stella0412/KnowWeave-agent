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
