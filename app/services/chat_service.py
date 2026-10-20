"""图驱动的聊天服务：准备会话 → 跑 LangGraph → 映射 SSE → 增量落 MySQL。"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from app.db.repository import ChatRepository
from app.graph.runtime import normalize_stream_event, thread_config
from app.graph.state import GraphContext
from app.rag.pipeline import TurnContext, default_source_files

_FIXED_REPLY_NODES = {"chitchat_reply", "complaint_reply", "fallback_reply"}


def _message_id(row) -> str:
    return f"db:{row.id}"


def _row_to_message(row) -> BaseMessage:
    kwargs = {"id": _message_id(row)}
    if row.role.value == "user":
        return HumanMessage(content=row.content or "", **kwargs)
    if row.role.value == "assistant":
        return AIMessage(
            content=row.content or "",
            tool_calls=row.tool_calls or [],
            **kwargs,
        )
    return ToolMessage(
        content=row.content or "",
        tool_call_id=row.tool_call_id or "",
        **kwargs,
    )


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content or "")


@dataclass
class _Prepared:
    conversation_id: int
    created: bool
    turn: TurnContext
    graph_input: dict
    history_count: int


class ChatService:
    def __init__(
        self,
        model,
        intent_model,
        session_factory,
        graph,
        *,
        token_budget: int = 4096,
        tool_timeout: float = 5.0,
        tool_retries: int = 1,
    ):
        self._model = model
        self._intent_model = intent_model
        self._session_factory = session_factory
        self._graph = graph
        self._token_budget = token_budget
        self._tool_timeout = tool_timeout
        self._tool_retries = tool_retries

    def _context(self, turn: TurnContext) -> GraphContext:
        return GraphContext(
            model=self._model,
            intent_model=self._intent_model,
            session_factory=self._session_factory,
            turn=turn,
            token_budget=self._token_budget,
            tool_timeout=self._tool_timeout,
            tool_retries=self._tool_retries,
        )

    async def _prepare(self, session_id: str | None, message: str) -> _Prepared:
        async with self._session_factory() as session:
            repo = ChatRepository(session)
            if session_id is None:
                conversation_id = await repo.create_conversation()
                await session.commit()
                created = True
            else:
                conversation_id = int(session_id)
                created = False
                if await repo.get_conversation(conversation_id) is None:
                    raise ValueError("会话不存在")
            user_row = await repo.add_message(conversation_id, role="user", content=message)
            await session.commit()
            new_user_id = user_row.id
            rows = await repo.list_messages(conversation_id)

        history_rows = [row for row in rows if row.id != new_user_id]
        config = thread_config(conversation_id)
        snapshot = await self._graph.aget_state(config)
        existing = list((snapshot.values or {}).get("messages") or []) if snapshot else []
        input_messages = [] if existing else [_row_to_message(row) for row in history_rows]
        input_messages.append(HumanMessage(content=message, id=f"db:{new_user_id}"))

        graph_input = {
            "messages": input_messages,
            "user_id": "anonymous",
            "conversation_id": conversation_id,
            "intent": None,
            "intent_source": None,
            "route": None,
            "evidence": None,
            "citations": [],
            "evidence_strong": None,
            "retrieval_error": None,
            "refusal_source": None,
            "refusal_reason": None,
            "answer": None,
            "steps": 0,
            "tokens_used": 0,
            "suggested_actions": [],
            "fallback_reason": None,
            "stop_reason": None,
            "trace": {"t0": time.perf_counter()},
        }
        turn = TurnContext(
            raw_question=message,
            conversation_id=conversation_id,
            source_files=default_source_files(),
        )
        # checkpoint 为空时，历史来自 MySQL 的补种消息；切片要跳过这段补种，
        # 否则 /chat/agent 会把上一轮的工具调用重复算进本轮载荷。
        history_count = len(existing) if existing else len(history_rows)
        return _Prepared(
            conversation_id=conversation_id,
            created=created,
            turn=turn,
            graph_input=graph_input,
            history_count=history_count,
        )

    async def _persist_new(self, conversation_id: int, state: dict, history_count: int) -> None:
        messages = list(state.get("messages") or [])
        new_messages = messages[history_count:]
        async with self._session_factory() as session:
            repo = ChatRepository(session)
            for message in new_messages:
                if isinstance(message, HumanMessage):
                    continue
                if str(getattr(message, "id", "") or "").startswith("db:"):
                    continue
                if isinstance(message, AIMessage):
                    await repo.add_message(
                        conversation_id,
                        role="assistant",
                        content=_text_of(message.content),
                        tool_calls=message.tool_calls or None,
                    )
                elif isinstance(message, ToolMessage):
                    await repo.add_message(
                        conversation_id,
                        role="tool",
                        content=_text_of(message.content),
                        tool_call_id=message.tool_call_id,
                    )
            await session.commit()

    async def stream(self, session_id: str | None, message: str):
        try:
            prepared = await self._prepare(session_id, message)
            if prepared.created:
                yield {"type": "session", "session_id": str(prepared.conversation_id)}

            config = thread_config(prepared.conversation_id)
            context = self._context(prepared.turn)
            async for raw_event in self._graph.astream(
                prepared.graph_input,
                config=config,
                context=context,
                stream_mode=["messages", "updates"],
            ):
                mode, payload = normalize_stream_event(raw_event)
                if mode == "messages":
                    chunk, metadata = payload
                    if (metadata or {}).get("langgraph_node") != "agent_llm":
                        continue
                    text = _text_of(getattr(chunk, "content", ""))
                    if text:
                        yield {"type": "token", "content": text}
                elif mode == "updates":
                    for node, update in _iter_node_updates(payload):
                        if node == "agent_llm":
                            for ai in _messages_of(update):
                                for call in getattr(ai, "tool_calls", None) or []:
                                    yield {
                                        "type": "tool_status",
                                        "name": call.get("name", ""),
                                        "status": "running",
                                    }
                        elif node in _FIXED_REPLY_NODES:
                            answer = (update or {}).get("answer")
                            if answer:
                                yield {"type": "token", "content": answer}

            snapshot = await self._graph.aget_state(config)
            state = dict(snapshot.values or {}) if snapshot else {}
            await self._persist_new(
                prepared.conversation_id, state, prepared.history_count
            )
            citations = state.get("citations") or prepared.turn.citations
            if citations:
                yield {"type": "citations", "items": citations}
            actions = state.get("suggested_actions") or []
            if actions:
                yield {"type": "actions", "items": actions}
            yield {"type": "done"}
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}

    async def run(self, session_id: str | None, message: str) -> dict:
        prepared = await self._prepare(session_id, message)
        config = thread_config(prepared.conversation_id)
        state = await self._graph.ainvoke(
            prepared.graph_input,
            config=config,
            context=self._context(prepared.turn),
        )
        state = dict(state or {})
        await self._persist_new(prepared.conversation_id, state, prepared.history_count)

        messages = list(state.get("messages") or [])
        new_messages = messages[prepared.history_count:]
        tool_calls = []
        tool_results = []
        for item in new_messages:
            if isinstance(item, AIMessage):
                tool_calls.extend(item.tool_calls or [])
            elif isinstance(item, ToolMessage):
                tool_results.append(
                    {"tool_call_id": item.tool_call_id, "content": _text_of(item.content)}
                )
        return {
            "session_id": prepared.conversation_id,
            "answer": state.get("answer"),
            "intent": state.get("intent"),
            "route": state.get("route"),
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "suggested_actions": state.get("suggested_actions") or [],
            "trace": state.get("trace") or {},
        }


def _iter_node_updates(payload):
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if isinstance(item, dict):
            for node, update in item.items():
                yield node, update


def _messages_of(update):
    if not isinstance(update, dict):
        return []
    return [m for m in update.get("messages") or [] if isinstance(m, AIMessage)]
