import pytest

from app.services.chat_service import ChatService


class EmptyGraph:
    async def aget_state(self, config):
        return None


@pytest.mark.asyncio
async def test_prepare_resets_all_turn_fields(session_factory):
    service = ChatService(
        model=object(), intent_model=object(), session_factory=session_factory,
        graph=EmptyGraph(),
    )
    prepared = await service._prepare(None, "这个能退吗")
    graph_input = prepared.graph_input
    for key in (
        "resolved_query", "coref_source", "intent_confidence", "order_no",
        "order_data", "order_error", "expand_degraded", "policy_refused",
        "refund_offered",
    ):
        assert graph_input[key] is None
    assert graph_input["policy_queries"] == []
    assert graph_input["flow_steps"] == []
    assert graph_input["intent"] is None
    assert graph_input["route"] is None
