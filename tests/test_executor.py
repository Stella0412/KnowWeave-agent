import asyncio

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from app.tools.executor import execute_tool


@tool
def echo(x: str) -> str:
    """Echo."""
    return f"echo:{x}"


@tool
def boom(x: str) -> str:
    """Boom."""
    raise RuntimeError("kaboom")


@tool
async def slow(x: str) -> str:
    """Slow."""
    await asyncio.sleep(10)
    return "done"


def _call(name, args, timeout=5.0, retries=0):
    by_name = {"echo": echo, "boom": boom, "slow": slow}
    return execute_tool(
        {"name": name, "id": "call_1", "args": args},
        by_name,
        timeout=timeout,
        retries=retries,
    )


@pytest.mark.asyncio
async def test_valid_tool_returns_result():
    tm = await _call("echo", {"x": "hi"})
    assert isinstance(tm, ToolMessage)
    assert tm.content == "echo:hi"
    assert tm.tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_invalid_args_returns_error_message():
    tm = await _call("echo", {"wrong": "hi"})
    assert "校验失败" in tm.content


@pytest.mark.asyncio
async def test_tool_raises_returns_error_message():
    tm = await _call("boom", {"x": "hi"})
    assert "执行失败" in tm.content


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_message():
    tm = await execute_tool({"name": "nope", "id": "c", "args": {}}, {}, timeout=1.0, retries=0)
    assert "未知工具" in tm.content


@pytest.mark.asyncio
async def test_timeout_returns_error_message():
    tm = await _call("slow", {"x": "hi"}, timeout=0.05, retries=0)
    assert "执行失败" in tm.content
