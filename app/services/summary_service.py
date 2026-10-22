"""后台摘要：单飞、冷却、冻结区间、追加事务。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.db.repository import ChatRepository
from app.graph.runtime import thread_config
from app.prompts.context import summary_messages
from app.services.context_manager import split_turns
from app.services.token_counter import text_of

logger = logging.getLogger("knowweave.context")


@dataclass(frozen=True)
class SummaryRange:
    from_msg_id: int
    upto_msg_id: int
    expected_summary_upto: int | None


def _log(event: str, **fields) -> None:
    logger.info(json.dumps({"event": event, "ts": time.time(), **fields}, ensure_ascii=False))


def _truncate_summary(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    index = max(cut.rfind("。"), cut.rfind("！"), cut.rfind("？"), cut.rfind("；"), cut.rfind("\n"))
    return (cut[: index + 1] if index >= 0 else cut).strip()


class SummaryService:
    def __init__(self, session_factory, model, graph, counter, settings, *, logger_=None):
        self._session_factory = session_factory
        self._model = model
        self._graph = graph
        self._counter = counter
        self._settings = settings
        self._log = logger_ or _log
        self._tasks: dict[int, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._cooldowns: dict[int, float] = {}

    async def schedule(self, conversation_id: int, frozen: SummaryRange) -> bool:
        async with self._lock:
            task = self._tasks.get(conversation_id)
            if task is not None and not task.done():
                self._log("summary skipped", reason="in_flight", conversation_id=conversation_id)
                return False
            until = self._cooldowns.get(conversation_id, 0.0)
            if time.monotonic() < until:
                self._log("summary skipped", reason="cooldown", conversation_id=conversation_id)
                return False
            task = asyncio.create_task(self._run(conversation_id, frozen))
            self._tasks[conversation_id] = task
            task.add_done_callback(
                lambda done: asyncio.create_task(self._forget(conversation_id, done))
            )
            return True

    async def _forget(self, conversation_id: int, task: asyncio.Task) -> None:
        async with self._lock:
            if self._tasks.get(conversation_id) is task:
                self._tasks.pop(conversation_id, None)

    async def wait_for_idle(self) -> None:
        while True:
            async with self._lock:
                tasks = [task for task in self._tasks.values() if not task.done()]
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown(self) -> None:
        async with self._lock:
            tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, conversation_id: int, frozen: SummaryRange) -> None:
        started = time.perf_counter()
        try:
            messages, source = await self._load_messages(conversation_id, frozen)
            _log(
                "summary start", conversation_id=conversation_id, source=source,
                from_msg_id=frozen.from_msg_id, upto_msg_id=frozen.upto_msg_id,
            )
            transcript = self._transcript(messages)
            result = await self._model.ainvoke(summary_messages(transcript))
            has_facts = bool(getattr(result, "has_facts", False))
            summary = _truncate_summary(
                getattr(result, "summary", "") or "", self._settings.summary_max_chars
            )
            if has_facts and not summary:
                raise ValueError("has_facts=True 但 summary 为空")
            seq = await self._commit(conversation_id, frozen, has_facts, summary)
            if not has_facts:
                _log(
                    "summary skipped", reason="no_facts", conversation_id=conversation_id,
                    from_msg_id=frozen.from_msg_id, upto_msg_id=frozen.upto_msg_id,
                )
                return
            _log(
                "summary done", conversation_id=conversation_id, seq=seq,
                from_msg_id=frozen.from_msg_id, upto_msg_id=frozen.upto_msg_id,
                chars=len(summary), elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._cooldowns[conversation_id] = (
                time.monotonic() + self._settings.summary_retry_cooldown_seconds
            )
            _log(
                "summary failed", conversation_id=conversation_id,
                from_msg_id=frozen.from_msg_id, upto_msg_id=frozen.upto_msg_id,
                error=type(exc).__name__, detail=str(exc)[:200],
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

    async def _load_messages(self, conversation_id: int, frozen: SummaryRange):
        messages: list[BaseMessage] = []
        try:
            snapshot = await self._graph.aget_state(thread_config(conversation_id))
            messages = list((getattr(snapshot, "values", None) or {}).get("messages") or [])
        except Exception:
            messages = []
        selected = self._slice(messages, frozen)
        if selected:
            return selected, "checkpoint"
        _log("summary source fallback", conversation_id=conversation_id, source="mysql")
        async with self._session_factory() as session:
            rows = await ChatRepository(session).list_messages(conversation_id)
        fallback = [
            HumanMessage(content=row.content or "", id=f"db:{row.id}")
            if row.role.value == "user"
            else AIMessage(content=row.content or "", id=f"db:{row.id}")
            for row in rows
            if frozen.from_msg_id <= row.id <= frozen.upto_msg_id
        ]
        return fallback, "mysql"

    def _slice(self, messages: list[BaseMessage], frozen: SummaryRange) -> list[BaseMessage]:
        selected: list[BaseMessage] = []
        for turn in split_turns(messages):
            anchor = turn.first_user_msg_id
            if anchor is None:
                continue
            if frozen.from_msg_id <= anchor <= frozen.upto_msg_id:
                selected.extend(turn.messages)
        return selected

    def _transcript(self, messages: list[BaseMessage]) -> str:
        lines: list[str] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                lines.append(f"用户：{text_of(message.content)}")
            elif isinstance(message, ToolMessage):
                content = self._counter.truncate_text(
                    text_of(message.content),
                    self._settings.summary_tool_result_max_tokens,
                    "…（工具结果已截断）",
                )
                lines.append(f"工具：{content}")
            elif isinstance(message, AIMessage):
                content = self._counter.truncate_text(
                    text_of(message.content),
                    self._settings.summary_message_max_tokens,
                    "…（客服消息已截断）",
                )
                if not content and message.tool_calls:
                    content = "[调用工具：" + "、".join(
                        call.get("name", "") for call in message.tool_calls
                    ) + "]"
                lines.append(f"客服：{content}")
        return "\n".join(lines)

    async def _commit(
        self, conversation_id: int, frozen: SummaryRange, has_facts: bool, summary: str
    ) -> int:
        async with self._session_factory() as session:
            repo = ChatRepository(session)
            conv = await repo.get_conversation_for_update(conversation_id)
            if conv is None:
                raise ValueError("会话不存在")
            rows = await repo.list_summaries(conversation_id)
            effective = conv.summary_upto_msg_id
            if effective is None and rows:
                effective = max(row.upto_msg_id for row in rows)
            if effective != frozen.expected_summary_upto:
                await session.rollback()
                _log(
                    "summary skipped", reason="stale", conversation_id=conversation_id,
                    expected=frozen.expected_summary_upto, actual=effective,
                )
                return 0
            seq = 0
            if has_facts:
                seq = await repo.next_summary_seq(conversation_id)
                await repo.add_summary(
                    conversation_id,
                    seq=seq,
                    from_msg_id=frozen.from_msg_id,
                    upto_msg_id=frozen.upto_msg_id,
                    content=summary,
                )
            conv.summary_upto_msg_id = frozen.upto_msg_id
            rows = await repo.list_summaries(conversation_id)
            projection = "\n".join(f"【第{row.seq}段】{row.content}" for row in rows)
            if len(projection) > 60000:
                projection = "【更早摘要见摘要表】\n" + projection[-59000:]
            conv.summary = projection
            await session.commit()
            return seq
