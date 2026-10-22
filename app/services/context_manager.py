"""会话上下文的分层规划器。

本文件先落纯函数；prepare/shared_history 在后续任务补齐。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.services.token_counter import text_of


@dataclass
class Turn:
    first_user_msg_id: int | None
    messages: list[BaseMessage] = field(default_factory=list)


def _db_id(message: BaseMessage) -> int | None:
    raw = getattr(message, "id", None)
    if isinstance(raw, str) and raw.startswith("db:"):
        try:
            return int(raw[3:])
        except ValueError:
            return None
    return None


def split_turns(messages: list[BaseMessage]) -> list[Turn]:
    """按用户消息切整轮；没有 db id 的旧消息就近并入下一个（否则上一个）锚点。"""
    messages = list(messages or [])
    next_ids: list[int | None] = [None] * len(messages)
    nxt: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage):
            anchor = _db_id(message)
            if anchor is not None:
                nxt = anchor
        next_ids[index] = nxt

    turns: list[Turn] = []
    current: Turn | None = None
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            anchor = _db_id(message)
            if anchor is None:
                anchor = next_ids[index]
                if anchor is None and turns:
                    anchor = turns[-1].first_user_msg_id
            if current is None or anchor is None or anchor != current.first_user_msg_id:
                current = Turn(first_user_msg_id=anchor, messages=[])
                turns.append(current)
        elif current is None:
            current = Turn(first_user_msg_id=None, messages=[])
            turns.append(current)
        current.messages.append(message)
    return turns


def classify_turn(
    turn: Turn, summary_upto: int | None, layer1_from: int | None
) -> int:
    anchor = turn.first_user_msg_id
    if anchor is None:
        return 1
    if summary_upto is not None and anchor <= summary_upto:
        return 3
    if layer1_from is not None and anchor >= layer1_from:
        return 1
    return 2


def render_layer2_turn(turn: Turn, prefix_chars: int = 60) -> list[BaseMessage]:
    rendered: list[BaseMessage] = []
    for message in turn.messages:
        if isinstance(message, HumanMessage):
            rendered.append(message.model_copy(deep=True))
        elif isinstance(message, AIMessage):
            text = text_of(message.content)
            if len(text) > prefix_chars:
                text = text[:prefix_chars] + "…"
            rendered.append(
                message.model_copy(deep=True, update={"content": text})
            )
        elif isinstance(message, ToolMessage):
            rendered.append(
                message.model_copy(
                    deep=True,
                    update={
                        "content": f"[工具结果已省略：{message.name or ''} {message.tool_call_id}]"
                    },
                )
            )
        else:
            rendered.append(message.model_copy(deep=True))
    return rendered


def serialize_message(message: BaseMessage) -> dict:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, ToolMessage):
        role = "tool"
    elif isinstance(message, SystemMessage):
        role = "system"
    else:
        role = getattr(message, "type", "unknown")
    return {
        "role": role,
        "content": text_of(getattr(message, "content", "")),
        "tool_calls": getattr(message, "tool_calls", None) or [],
        "tool_call_id": getattr(message, "tool_call_id", None),
        "name": getattr(message, "name", None),
        "id": getattr(message, "id", None),
    }
