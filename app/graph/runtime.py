"""图运行时的小工具：thread_id、流事件形状适配、checkpointer 生命周期。"""

from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def thread_config(conversation_id: int, *, turn_token: str | None = None) -> dict:
    config = {"configurable": {"thread_id": str(conversation_id)}}
    if turn_token:
        # 只有本轮显式带 token 时才加 metadata；无 token 的旧调用保持原形状，
        # 既让 ch05 的精确断言继续成立，也避免历史快照查询多带无意义字段。
        config["metadata"] = {
            "conversation_id": conversation_id,
            "turn_token": turn_token,
        }
    return config


def normalize_stream_event(event) -> tuple[str, object]:
    """兼容 LangGraph v1 的 (mode, payload) 与 v2 的 {"type","data"}。"""
    if isinstance(event, tuple) and len(event) == 2:
        return event[0], event[1]
    if isinstance(event, dict) and "type" in event and "data" in event:
        return event["type"], event["data"]
    raise ValueError(f"无法识别的流事件: {event!r}")


@asynccontextmanager
async def open_checkpointer(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        await saver.setup()
        yield saver
