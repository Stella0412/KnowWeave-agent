from app.graph.routing import INTENTS, ROUTE_BY_INTENT


def test_eight_intents_map_to_six_routes():
    assert set(INTENTS) == set(ROUTE_BY_INTENT)
    assert set(ROUTE_BY_INTENT.values()) == {
        "business", "knowledge", "refund_flow", "complaint", "chitchat", "clarify",
    }
    assert ROUTE_BY_INTENT["refund_return"] == "refund_flow"
    assert ROUTE_BY_INTENT["after_sale"] == "refund_flow"
    assert ROUTE_BY_INTENT["other"] == "clarify"
