from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from app.observability.spans import tool_span
from app.tools.contracts import ExecutionContext, ToolDefinition, ToolOutcome
from app.tools.executor import ToolEngine


class FakeSpan:
    def __init__(self):
        self.updates = []
        self.ended = False

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended = True


class FakeClient:
    def __init__(self):
        self.span = FakeSpan()
        self.calls = []

    def start_observation(self, **kwargs):
        self.calls.append(kwargs)
        return self.span


class Policy:
    def load(self):
        return SimpleNamespace(permits=lambda definition: True)


def _definition(invoke):
    return ToolDefinition(
        "demo", "demo", {"type": "object", "properties": {}},
        "builtin", None, None, invoke,
    )


async def test_tool_span_success_and_failure():
    client = FakeClient()
    with tool_span(client, "demo", input={"x": "y"}, metadata={"turn_token": "t"}) as span:
        span.update(output="ok", level="DEFAULT")
    assert client.calls[0]["name"] == "demo"
    assert client.calls[0]["as_type"] == "tool"
    assert client.span.ended is True

    client2 = FakeClient()
    with pytest.raises(RuntimeError):
        with tool_span(client2, "demo", input={}, metadata={}):
            raise RuntimeError("boom")
    assert client2.span.updates[-1]["level"] == "ERROR"


async def test_engine_calls_tracer_and_truncates():
    client = FakeClient()
    seen = []

    def tracer(name, *, input, metadata):
        seen.append((name, input, metadata))
        return tool_span(client, name, input=input, metadata=metadata)

    async def invoke(args, context):
        return ToolOutcome("success", data={"ok": True})

    engine = ToolEngine(Policy(), lambda record: None, timeout=5.0, retries=0, tracer=tracer)
    ctx = ExecutionContext(
        conversation_id=1, tool_call_id="c1", turn_token="t1",
    )
    message = await engine.execute(
        {"name": "demo", "id": "c1", "args": {}}, ctx,
        {"demo": _definition(invoke)},
    )
    assert isinstance(message, ToolMessage)
    assert seen[0][0] == "demo"
    assert seen[0][2]["turn_token"] == "t1"
    assert client.span.updates[-1]["level"] == "DEFAULT"


async def test_engine_without_tracer_is_noop():
    async def invoke(args, context):
        return ToolOutcome("success", data="ok")

    engine = ToolEngine(Policy(), lambda record: None, timeout=5.0, retries=0)
    message = await engine.execute(
        {"name": "demo", "id": "c1", "args": {}}, ExecutionContext(1, "c1"),
        {"demo": _definition(invoke)},
    )
    assert isinstance(message, ToolMessage)
