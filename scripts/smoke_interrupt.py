import asyncio
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class S(TypedDict, total=False):
    picked: str


def pick(state):
    return {"picked": interrupt({"kind": "pick", "items": [1, 2]})}


def main():
    builder = StateGraph(S)
    builder.add_node("pick", pick)
    builder.add_edge(START, "pick")
    builder.add_edge("pick", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "smoke"}}

    async def run():
        first = await graph.ainvoke({}, config=config)
        print("INVOKE_KEYS", sorted(first.keys()))
        snapshot = await graph.aget_state(config)
        print("TASKS", [
            (getattr(t, "name", None), getattr(t, "interrupts", None))
            for t in getattr(snapshot, "tasks", ()) or ()
        ])
        second = await graph.ainvoke(Command(resume="1001"), config=config)
        print("RESUMED", second)

    asyncio.run(run())


if __name__ == "__main__":
    main()
