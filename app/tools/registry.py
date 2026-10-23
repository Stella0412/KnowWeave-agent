import copy

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
from app.tools.contracts import ToolDefinition
from app.tools.validation import check_schema

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


class ToolRegistry:
    """动态工具注册中心。

    与固定 build_tools 并存（Task 6 移除固定绑定）。register 后下一次 snapshot 即包含；
    重名显式报错不覆盖；snapshot 返回完全独立的深拷贝，注册后外部改动定义或改动快照
    都不影响注册表或其他快照。
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        check_schema(definition.parameters)  # 非法 Schema 永不入册
        if definition.name in self._tools:
            raise ValueError(f"工具名重复: {definition.name}")
        self._tools[definition.name] = copy.deepcopy(definition)

    def snapshot(self) -> dict[str, ToolDefinition]:
        return {name: copy.deepcopy(tool) for name, tool in self._tools.items()}
