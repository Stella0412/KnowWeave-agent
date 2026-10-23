"""把确定性节点与 ReAct 环组装成 LangGraph 图。"""

from langgraph.graph import END, START, StateGraph

from app.graph.agent import agent_llm, agent_tools
from app.graph.nodes import (
    chitchat_reply,
    classify_intent,
    complaint_reply,
    confidence_check,
    coref,
    fallback_reply,
    forced_rag,
    log_turn,
    prepare_context,
    refund_expand,
    refund_fetch_order,
    refund_retrieve,
)
from app.graph.routing import (
    gate_router,
    make_should_continue,
    make_tool_loop_router,
    refund_order_router,
    route_by_intent,
)
from app.graph.state import ConversationState, GraphContext


def build_graph(*, checkpointer, max_agent_steps: int = 6):
    graph = StateGraph(ConversationState, context_schema=GraphContext)

    graph.add_node("prepare_context", prepare_context)
    graph.add_node("coref", coref)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("forced_rag", forced_rag)
    graph.add_node("confidence_check", confidence_check)
    graph.add_node("refund_fetch_order", refund_fetch_order)
    graph.add_node("refund_expand", refund_expand)
    graph.add_node("refund_retrieve", refund_retrieve)
    graph.add_node("agent_llm", agent_llm)
    graph.add_node("agent_tools", agent_tools)
    graph.add_node("fallback_reply", fallback_reply)
    graph.add_node("complaint_reply", complaint_reply)
    graph.add_node("chitchat_reply", chitchat_reply)
    graph.add_node("log", log_turn)

    graph.add_edge(START, "prepare_context")
    graph.add_edge("prepare_context", "coref")
    graph.add_edge("coref", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "knowledge": "forced_rag",
            "business": "agent_llm",
            "complaint": "complaint_reply",
            "chitchat": "chitchat_reply",
            "clarify": "agent_llm",
            "refund_flow": "refund_fetch_order",
        },
    )
    graph.add_edge("forced_rag", "confidence_check")
    graph.add_conditional_edges(
        "confidence_check",
        gate_router,
        {"agent": "agent_llm", "fallback": "fallback_reply"},
    )
    graph.add_conditional_edges(
        "refund_fetch_order",
        refund_order_router,
        {"expand": "refund_expand", "fallback": "fallback_reply"},
    )
    graph.add_edge("refund_expand", "refund_retrieve")
    graph.add_edge("refund_retrieve", "confidence_check")
    graph.add_conditional_edges(
        "agent_llm",
        make_should_continue(max_agent_steps),
        {"tools": "agent_tools", "log": "log", "fallback": "fallback_reply"},
    )
    graph.add_conditional_edges(
        "agent_tools",
        make_tool_loop_router(),
        {"tools": "agent_tools", "agent_llm": "agent_llm"},
    )
    graph.add_edge("fallback_reply", "log")
    graph.add_edge("complaint_reply", "log")
    graph.add_edge("chitchat_reply", "log")
    graph.add_edge("log", END)

    return graph.compile(checkpointer=checkpointer)
