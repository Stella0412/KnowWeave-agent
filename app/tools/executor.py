import asyncio

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool


async def execute_tool(
    tool_call: dict,
    tools_by_name: dict[str, BaseTool],
    *,
    timeout: float,
    retries: int,
) -> ToolMessage:
    """执行单个工具调用：参数校验 → 超时重试 → 错误兜底，永远返回 ToolMessage 不抛。"""
    name = tool_call.get("name", "")
    call_id = tool_call.get("id", "")
    args = tool_call.get("args") or {}

    tool = tools_by_name.get(name)
    if tool is None:
        return ToolMessage(content=f"未知工具: {name}", tool_call_id=call_id, name=name)

    # 1. 参数 Schema 校验（args_schema 为 Pydantic 模型）
    if tool.args_schema is not None:
        try:
            tool.args_schema.model_validate(args)
        except Exception as e:
            return ToolMessage(
                content=f"工具参数校验失败: {e}", tool_call_id=call_id, name=name
            )

    # 2. 超时 + 重试执行
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
            content = result if isinstance(result, str) else str(result)
            return ToolMessage(content=content, tool_call_id=call_id, name=name)
        except Exception as e:  # 超时或工具自身报错都进兜底
            last_err = e

    return ToolMessage(
        content=f"工具执行失败: {last_err}", tool_call_id=call_id, name=name
    )
