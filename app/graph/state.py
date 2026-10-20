"""ch05 图的 State 与每轮运行时依赖。

State 进 checkpointer；GraphContext 不进 checkpointer，由 ChatService 每轮注入。
"""

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.rag.pipeline import TurnContext


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    conversation_id: int

    intent: str | None
    intent_source: str | None
    route: str | None

    evidence: str | None
    citations: list[dict]
    evidence_strong: bool | None
    retrieval_error: str | None
    refusal_source: str | None
    refusal_reason: str | None

    answer: str | None
    steps: int
    tokens_used: int
    suggested_actions: list[dict]
    fallback_reason: str | None
    stop_reason: str | None
    trace: dict


@dataclass
class GraphContext:
    model: Any
    intent_model: Any
    session_factory: Any
    turn: TurnContext
    token_budget: int = 4096
    tool_timeout: float = 5.0
    tool_retries: int = 1
