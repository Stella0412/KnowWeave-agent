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


def tool_timeout(name: str, default: float) -> float:
    return settings.rag_tool_timeout if name in _LONG_RUNNING_TOOLS else default


def build_tools(
    session_factory,
    conversation_id: int,
    turn: TurnContext | None = None,
    *,
    include_faq: bool = True,
    include_lookup: bool = True,
    include_refund: bool = False,
) -> list[BaseTool]:
    tools: list[BaseTool] = []
    if include_lookup:
        tools.extend([build_query_order(session_factory), query_product, query_logistics])
    if include_faq:
        tools.append(build_query_faq(session_factory, turn))
    if include_refund:
        tools.append(submit_refund)
    tools.append(build_create_ticket(session_factory, conversation_id))
    return tools
