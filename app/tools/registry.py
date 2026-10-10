from langchain_core.tools import BaseTool

from app.tools.business import (
    build_create_ticket,
    build_query_faq,
    query_logistics,
    query_order,
    query_product,
)


def build_tools(session_factory, conversation_id: int) -> list[BaseTool]:
    return [
        query_order,
        query_product,
        query_logistics,
        build_query_faq(session_factory),
        build_create_ticket(session_factory, conversation_id),
    ]
