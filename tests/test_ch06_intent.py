from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.graph.nodes import classify_intent
from app.graph.routing import INTENTS, ROUTE_BY_INTENT
from app.prompts.orchestrator import IntentOutput, intent_messages


class FakeIntentModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.seen = []

    async def ainvoke(self, messages):
        self.seen.append(messages)
        if self.error:
            raise self.error
        return self.result


def _runtime(model):
    return SimpleNamespace(context=SimpleNamespace(intent_model=model))


@pytest.mark.asyncio
async def test_intent_sets_confidence_and_refund_route():
    model = FakeIntentModel(IntentOutput(intent="refund_return", confidence=0.93))
    update = await classify_intent(
        {"resolved_query": "猫爬架能退吗", "messages": [HumanMessage(content="这个能退吗")]},
        _runtime(model),
    )
    assert update == {
        "intent": "refund_return",
        "intent_confidence": 0.93,
        "route": "refund_flow",
        "intent_source": "model",
    }


@pytest.mark.asyncio
async def test_intent_fallback_is_other_not_after_sale():
    model = FakeIntentModel(error=RuntimeError("llm down"))
    update = await classify_intent(
        {"resolved_query": "奇怪的问题", "messages": [HumanMessage(content="奇怪的问题")]},
        _runtime(model),
    )
    assert update["intent"] == "other"
    assert update["route"] == "clarify"
    assert update["intent_source"] == "fallback"
    assert update["intent_confidence"] is None


@pytest.mark.asyncio
async def test_intent_rejects_unknown_label_and_bad_confidence():
    for result in (
        SimpleNamespace(intent="unknown", confidence=0.9),
        SimpleNamespace(intent="other", confidence=1.5),
        SimpleNamespace(intent="other", confidence="high"),
    ):
        update = await classify_intent(
            {"resolved_query": "x", "messages": [HumanMessage(content="x")]},
            _runtime(FakeIntentModel(result)),
        )
        assert update["intent"] == "other"
        assert update["intent_source"] == "fallback"


def test_intent_prompt_mentions_all_nine_labels():
    text = intent_messages("测试")[0].content
    for label in INTENTS:
        assert label in text
