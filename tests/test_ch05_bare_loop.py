import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from scripts.bare_agent_loop import run_bare_agent


class ScriptedModel:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.seen = []

    def bind_tools(self, tools):
        self.bound_tools = [t.name for t in tools]
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.seen.append(list(messages))
        return self._responses.pop(0)


@tool
async def echo(text: str) -> str:
    """原样返回。"""
    return f"echo:{text}"


@pytest.mark.asyncio
async def test_bare_loop_converges_after_one_tool():
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {"text": "hi"}, "id": "call_1", "type": "tool_call"}
        ]),
        AIMessage(content="done"),
    ])
    answer = await run_bare_agent(
        model, {"echo": echo}, [HumanMessage(content="please echo")], max_steps=4
    )
    assert answer.content == "done"
    assert model.calls == 2
    assert model.bound_tools == ["echo"]
    assert isinstance(model.seen[1][-1], ToolMessage)
    assert model.seen[1][-1].content == "echo:hi"


@pytest.mark.asyncio
async def test_bare_loop_stops_at_max_steps():
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {"text": "1"}, "id": "c1", "type": "tool_call"}
        ]),
        AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {"text": "2"}, "id": "c2", "type": "tool_call"}
        ]),
    ])
    answer = await run_bare_agent(
        model, {"echo": echo}, [HumanMessage(content="loop")], max_steps=2
    )
    assert "最大步数" in answer.content
    assert model.calls == 2
