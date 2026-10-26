"""不依赖 LangGraph 的裸 Agent 循环：模型 → 工具 → 模型，直到不再要工具。

这一版故意只保留最小控制流，用来确认「Agent 就是一个带停止条件的工具循环」；
ch05 的生产实现只是把这个循环拆成图里的 `agent_llm ↔ agent_tools`。
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool


async def run_bare_agent(model, tools_by_name, messages, max_steps=4):
    messages = list(messages)
    bound = model.bind_tools(list(tools_by_name.values()))
    for _ in range(max_steps):
        ai = await bound.ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return ai
        for tc in ai.tool_calls:
            tool_obj = tools_by_name[tc["name"]]
            result = await tool_obj.ainvoke(tc["args"])
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tc.get("id", ""),
                    name=tc["name"],
                )
            )
    return AIMessage(content=f"已达到最大步数 {max_steps}，先停在这里。")


class _DemoModel:
    """演示用脚本模型：第一次要工具，第二次收敛。"""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "echo", "args": {"text": "你好"}, "id": "c1", "type": "tool_call"}
                ],
            )
        return AIMessage(content="裸循环已收敛：工具结果已经回灌。")


@tool
async def echo(text: str) -> str:
    """原样返回。"""
    return f"echo:{text}"


async def main():
    model = _DemoModel()
    answer = await run_bare_agent(
        model, {"echo": echo}, [HumanMessage(content="演示")], max_steps=4
    )
    print(f"模型调用次数: {model.calls}")
    print(f"最终回答: {answer.content}")


if __name__ == "__main__":
    asyncio.run(main())
