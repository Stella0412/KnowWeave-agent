from langchain_core.tools import BaseTool

from app.config import settings
from app.rag.pipeline import TurnContext
from app.tools.business import (
    build_create_ticket,
    build_query_faq,
    build_query_order,
    query_logistics,
    query_product,
    submit_refund,
)

# 单轮耗时的工具要单独给预算：query_faq 一轮要跑改写(LLM) + 嵌入 + 混合检索 + 精排(网络)，
# 5 秒的全局 tool_timeout 会把它当场掐断（还白重试一次）。
_LONG_RUNNING_TOOLS = {"query_faq"}

ROUTE_TOOL_WHITELIST: dict[object, set[str]] = {
    "knowledge": {"query_order", "query_product", "query_logistics", "create_ticket"},
    "business": {
        "query_order", "query_product", "query_logistics", "query_faq", "create_ticket",
    },
    "clarify": {
        "query_order", "query_product", "query_logistics", "query_faq", "create_ticket",
    },
    ("refund_flow", "refund_return"): {"submit_refund", "create_ticket"},
    ("refund_flow", "after_sale"): {"create_ticket"},
}


def allowed_tool_names(route: str, intent: str | None) -> set[str]:
    if route == "refund_flow":
        return set(ROUTE_TOOL_WHITELIST.get((route, intent), {"create_ticket"}))
    return set(ROUTE_TOOL_WHITELIST.get(route, ROUTE_TOOL_WHITELIST["business"]))


def tool_timeout(name: str, default: float) -> float:
    return settings.rag_tool_timeout if name in _LONG_RUNNING_TOOLS else default


def build_tools(
    session_factory,
    conversation_id: int,
    turn: TurnContext | None = None,
) -> list[BaseTool]:
    """固定顺序、固定 schema；路由差异只由执行端白名单决定。"""
    return [
        build_query_order(session_factory),
        query_product,
        query_logistics,
        build_query_faq(session_factory, turn),
        submit_refund,
        build_create_ticket(session_factory, conversation_id),
    ]
