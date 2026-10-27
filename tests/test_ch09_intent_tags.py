"""意图标签：classify_intent 之后的观测要带上 intent:<x>，供 Metrics API 按 tags 归组。"""

from typing import TypedDict
from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langfuse import Langfuse, propagate_attributes
from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.resource_manager import LangfuseResourceManager
from langgraph.graph import END, START, StateGraph
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from app.observability.intent_tags import (
    IntentTaggingCallbackHandler,
    extract_intent,
    merge_span_tags,
)

TAGS = LangfuseOtelSpanAttributes.TRACE_TAGS


class _FakeSpan:
    def __init__(self, attributes=None):
        self.attributes = dict(attributes or {})

    def set_attribute(self, key, value):
        self.attributes[key] = tuple(value)


def _tags(span) -> set[str]:
    return set(span.attributes.get(TAGS) or ())


def test_extract_intent_accepts_known_label():
    assert extract_intent({"intent": "logistics"}) == "logistics"
    assert extract_intent({"intent": "refund_return"}) == "refund_return"


@pytest.mark.parametrize(
    "outputs",
    [None, {}, {"intent": "weather"}, {"intent": 3}, "logistics", ["logistics"]],
)
def test_extract_intent_ignores_unusable_outputs(outputs):
    assert extract_intent(outputs) is None


def test_merge_span_tags_keeps_existing_and_dedupes():
    span = _FakeSpan({TAGS: ("chat", "mode:chat")})
    assert merge_span_tags(span, "intent:logistics") is True
    assert span.attributes[TAGS] == ("chat", "mode:chat", "intent:logistics")
    assert merge_span_tags(span, "intent:logistics") is False
    assert span.attributes[TAGS].count("intent:logistics") == 1


def test_merge_span_tags_without_existing_tags():
    span = _FakeSpan()
    assert merge_span_tags(span, "intent:order") is True
    assert span.attributes[TAGS] == ("intent:order",)


class _State(TypedDict, total=False):
    intent: str
    answer: str


def _build_graph():
    async def classify_intent(state):
        return {"intent": "logistics"}

    async def answer(state, config):
        model = FakeListChatModel(responses=["好的"])
        await model.ainvoke([HumanMessage(content="我的快递到哪了")], config)
        return {"answer": "好的"}

    builder = StateGraph(_State)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("answer", answer)
    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


@pytest.fixture
def in_memory_client():
    exporter = InMemorySpanExporter()
    public_key = f"pk-intent-tags-{uuid4().hex}"
    client = Langfuse(
        public_key=public_key,
        secret_key="sk-test",
        host="http://localhost:3000",
        tracing_enabled=True,
        span_exporter=exporter,
    )
    try:
        yield client, exporter, public_key
    finally:
        try:
            client.flush()
        finally:
            LangfuseResourceManager.reset()


async def test_observations_after_classify_carry_intent_tag(in_memory_client):
    client, exporter, public_key = in_memory_client
    graph = _build_graph()
    handler = IntentTaggingCallbackHandler(public_key=public_key)

    with client.start_as_current_observation(
        as_type="span", name="handle-chat-turn"
    ) as root:
        with propagate_attributes(
            tags=["chat", "mode:test"],
            session_id="1",
            trace_name="handle-chat-turn",
        ):
            await graph.ainvoke({}, config={"callbacks": [handler]})

    client.flush()
    spans = exporter.get_finished_spans()
    by_name = {}
    for span in spans:
        by_name.setdefault(span.name, []).append(span)

    # classify_intent 自身的链不带意图标签（本轮标签在此之后才确定）
    classify_spans = by_name["classify_intent"]
    assert len(classify_spans) == 1
    assert _tags(classify_spans[0]) == {"chat", "mode:test"}

    # 分类之后的节点与 LLM 生成都带 intent:logistics，且保留原有 chat/mode
    answer_spans = by_name["answer"]
    assert len(answer_spans) == 1
    assert _tags(answer_spans[0]) == {"chat", "mode:test", "intent:logistics"}

    generations = [
        span
        for span in spans
        if span.attributes.get(LangfuseOtelSpanAttributes.OBSERVATION_TYPE)
        == "generation"
    ]
    assert len(generations) == 1
    assert _tags(generations[0]) == {"chat", "mode:test", "intent:logistics"}

    # 根 span 也补上意图标签，trace 列表能看到
    root_spans = by_name["handle-chat-turn"]
    assert len(root_spans) == 1
    assert _tags(root_spans[0]) == {"chat", "mode:test", "intent:logistics"}

    # 两条观测同属一条 trace
    assert len({span.context.trace_id for span in spans}) == 1
