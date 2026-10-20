"""图节点。本 Task 只落意图分类需要的部分，其余节点在后续 Task 补齐。"""

import logging

from langchain_core.messages import HumanMessage

from app.graph.routing import ROUTE_BY_INTENT
from app.prompts.orchestrator import intent_messages

logger = logging.getLogger("knowweave.turn")


def _last_user_text(messages) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


async def classify_intent(state, runtime):
    question = _last_user_text(state.get("messages", []))
    try:
        result = await runtime.context.intent_model.ainvoke(intent_messages(question))
        intent = getattr(result, "intent", None)
        if intent not in ROUTE_BY_INTENT:
            raise ValueError(f"未知意图: {intent!r}")
        source = "model"
    except Exception:
        logger.warning("意图分类失败，兜底 after_sale", exc_info=True)
        intent = "after_sale"
        source = "fallback"
    return {
        "intent": intent,
        "route": ROUTE_BY_INTENT[intent],
        "intent_source": source,
    }
