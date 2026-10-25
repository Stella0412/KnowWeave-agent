from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.graph.nodes import classify_intent
from app.graph.routing import INTENTS, ROUTE_BY_INTENT, route_by_intent
from app.prompts.orchestrator import IntentOutput, intent_messages


class FakeIntentModel:
    def __init__(self, intent=None, error=None):
        self.intent = intent
        self._error = error
        self.seen = []

    async def ainvoke(self, messages):
        self.seen.append(messages)
        if self._error:
            raise self._error
        return IntentOutput(intent=self.intent, confidence=0.9)


def _runtime(intent_model):
    return SimpleNamespace(context=SimpleNamespace(intent_model=intent_model))


def test_route_mapping_covers_all_nine_intents():
    assert set(INTENTS) == {
        "logistics", "order", "product_consult", "refund_return",
        "after_sale", "complaint", "human", "chitchat", "other",
    }
    assert ROUTE_BY_INTENT == {
        "logistics": "business",
        "order": "business",
        "product_consult": "knowledge",
        "refund_return": "refund_flow",
        "after_sale": "refund_flow",
        "complaint": "complaint",
        "human": "business",
        "chitchat": "chitchat",
        "other": "clarify",
    }


def test_route_by_intent_reads_state_route():
    assert route_by_intent({"route": "knowledge"}) == "knowledge"


@pytest.mark.asyncio
async def test_classify_intent_sets_intent_route_and_source():
    model = FakeIntentModel("refund_return")
    state = {"messages": [HumanMessage(content="拆封了还能退吗")]}
    update = await classify_intent(state, _runtime(model))
    assert update == {
        "intent": "refund_return",
        "intent_confidence": 0.9,
        "route": "refund_flow",
        "intent_source": "model",
    }
    assert "拆封了还能退吗" in model.seen[0][-1].content


@pytest.mark.asyncio
async def test_classify_intent_falls_back_to_other():
    model = FakeIntentModel(error=RuntimeError("llm down"))
    state = {"messages": [HumanMessage(content="你好")]}
    update = await classify_intent(state, _runtime(model))
    assert update["intent"] == "other"
    assert update["route"] == "clarify"
    assert update["intent_source"] == "fallback"
    assert update["intent_confidence"] is None


@pytest.mark.asyncio
async def test_classify_intent_rejects_unknown_label():
    model = FakeIntentModel("unknown")
    state = {"messages": [HumanMessage(content="你好")]}
    update = await classify_intent(state, _runtime(model))
    assert update["intent"] == "other"
    assert update["intent_source"] == "fallback"


def test_intent_prompt_mentions_all_labels():
    text = intent_messages("测试")[0].content
    for label in INTENTS:
        assert label in text
