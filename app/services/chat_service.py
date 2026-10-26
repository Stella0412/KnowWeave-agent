"""图驱动的聊天服务：准备会话 → 跑 LangGraph → 映射 SSE → 增量落 MySQL。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from weakref import WeakValueDictionary

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.types import Command

from app.config import settings as default_settings
from app.db.repository import ChatRepository, OrderRepository
from app.graph.runtime import normalize_stream_event, thread_config
from app.graph.state import GraphContext
from app.observability.context import TurnObserver, observer_scope
from app.rag.evidence import cap_citations
from app.rag.pipeline import TurnContext, default_source_files
from app.services.context_manager import ContextManager
from app.services.summary_service import SummaryService
from app.services.token_counter import TokenCounter, compute_budget
from app.tools.contracts import ExecutionContext
from app.tools.ticket_authorization import update_ticket_request
from app.tools.ticket_confirmation import confirm_ticket_call, draft_from_call, ticket_no_for

logger = logging.getLogger("knowweave.turn")

_FIXED_REPLY_NODES = {"chitchat_reply", "complaint_reply", "fallback_reply"}

# 每个会话一把异步锁：同一会话的发送与恢复串行化，不同会话互不阻塞。
# WeakValueDictionary 保证锁随会话不再被引用而释放。
_conversation_locks: WeakValueDictionary = WeakValueDictionary()
_conversation_locks_guard = asyncio.Lock()


class TicketResumeRejected(Exception):
    """工单恢复请求在校验阶段被拒绝；携带映射到 HTTP 的状态码。"""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class TicketEngineFault(RuntimeError):
    """旧 /chat/ticket 按钮的建单引擎内部故障（未产出成功结果），映射到 5xx。"""


class _ResumeLock:
    """校验阶段持有的会话锁：SSE 流接管后由流 finally 释放；若流从未被迭代，则由
    SSE 响应的 finally 兜底释放，避免锁被永久持有（会话级串行化不能因此死锁）。"""

    __slots__ = ("_held_locks", "conversation_id", "lock", "claimed")

    def __init__(self, held_locks: dict, conversation_id: int, lock: asyncio.Lock) -> None:
        self._held_locks = held_locks
        self.conversation_id = conversation_id
        self.lock = lock
        self.claimed = False

    def claim(self) -> asyncio.Lock:
        """SSE 流开始接管锁；此后由流的 finally 负责释放。"""
        self.claimed = True
        return self.lock

    def release_if_unclaimed(self) -> None:
        """仅当流从未接管时释放；已接管则由流自身释放，这里是无操作。"""
        if self.claimed:
            return
        if self._held_locks.get(self.conversation_id) is self:
            self._held_locks.pop(self.conversation_id, None)
        self.lock.release()


async def _lock_for_conversation(conversation_id: int) -> asyncio.Lock:
    async with _conversation_locks_guard:
        lock = _conversation_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            _conversation_locks[conversation_id] = lock
        return lock


async def _auto_approve_confirmation(definition, arguments, context) -> bool:
    """旧 /chat/ticket 按钮的服务端内部确认：跳过 interrupt，直接视为用户确认。"""
    context.ticket_decision = "confirm"
    context.confirmation_reason = None
    return True


def _interrupt_value(snapshot):
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            return getattr(item, "value", item)
    return None


def _interrupt_from_result(result):
    pending = (result or {}).get("__interrupt__")
    if not pending:
        return None
    first = pending[0] if isinstance(pending, (list, tuple)) else pending
    return getattr(first, "value", first)


def _interrupt_actions(payload) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    if payload.get("kind") == "select_order":
        return [{
            "type": "select_order",
            "label": "选择订单",
            "prompt": payload.get("prompt") or "请选择要退款的订单",
            "orders": list(payload.get("orders") or []),
        }]
    if payload.get("type") == "ticket_confirmation":
        draft = payload.get("draft") or {}
        return [{
            "type": "ticket_preview",
            "label": "确认建单",
            "confirmation_id": payload.get("confirmation_id"),
            "draft": {
                "ticket_type": draft.get("ticket_type"),
                "description": draft.get("description"),
            },
        }]
    return []


def _last_human_text(messages) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return _text_of(message.content)
    return ""


def _message_id(row) -> str:
    return f"db:{row.id}"


def _row_to_seed_message(row) -> BaseMessage | None:
    if row.role.value == "user":
        return HumanMessage(content=row.content or "", id=_message_id(row))
    if row.role.value == "assistant" and (row.content or "").strip():
        # 旧 assistant 的 tool_calls 不 seed：没有配对的 ToolMessage 会是协议错误
        return AIMessage(content=row.content or "", id=_message_id(row))
    return None


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
    user_id: str = "anonymous"


@dataclass
class _TurnRun:
    conversation_id: int
    token: str
    started: float
    config: dict
    finished: bool = False


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
        coref_model=None,
        expansion_model=None,
        intent_mode: str = "accuracy",
        settings=None,
        context_manager: ContextManager | None = None,
        summary_service: SummaryService | None = None,
        counter: TokenCounter | None = None,
        registry=None,
        provider=None,
        engine=None,
        metrics_handler=None,
        turn_metrics_writer=None,
        trace_scope=None,
    ):
        self._settings = settings or default_settings
        self._counter = counter or TokenCounter(
            cjk_tokens_per_char=self._settings.cjk_tokens_per_char,
            estimate_factor=self._settings.token_estimate_factor,
        )
        self._budget = compute_budget(self._settings)
        self._model = model
        self._intent_model = intent_model
        self._session_factory = session_factory
        self._graph = graph
        self._token_budget = token_budget
        self._tool_timeout = tool_timeout
        self._tool_retries = tool_retries
        self._coref_model = coref_model
        self._expansion_model = expansion_model
        self._intent_mode = intent_mode
        self._context_manager = context_manager
        self._summary_service = summary_service
        self._registry, self._provider, self._engine = self._resolve_tool_runtime(
            registry, provider, engine,
        )
        self._metrics_handler = metrics_handler
        self._turn_metrics_writer = turn_metrics_writer
        self._trace_scope = trace_scope
        # SSE 恢复路由里，validate 阶段先取锁并暂存，stream 生成器接管后在 finally 释放，
        # 让锁覆盖「校验 + 流式完成/断开」全过程。键为 conversation_id。
        self._held_locks: dict[int, _ResumeLock] = {}

    def _resolve_tool_runtime(self, registry, provider, engine):
        from app.tools.audit import write_audit
        from app.tools.builtin import register_builtins
        from app.tools.executor import ToolEngine
        from app.tools.policy import PolicyStore
        from app.tools.registry import ToolRegistry, tool_timeout

        if registry is None:
            registry = ToolRegistry()
            register_builtins(registry)
        if engine is None:
            policy = PolicyStore(self._settings.tool_permissions_path)
            engine = ToolEngine(
                policy,
                lambda record: write_audit(self._session_factory, record),
                timeout=lambda name: tool_timeout(name, self._settings.tool_timeout),
                retries=self._settings.tool_retries,
            )
        return registry, provider, engine

    def _context(self, turn: TurnContext, turn_token: str | None = None) -> GraphContext:
        return GraphContext(
            model=self._model,
            intent_model=self._intent_model,
            coref_model=self._coref_model,
            expansion_model=self._expansion_model,
            intent_mode=self._intent_mode,
            session_factory=self._session_factory,
            turn=turn,
            token_budget=self._token_budget,
            tool_timeout=self._tool_timeout,
            tool_retries=self._tool_retries,
            context_manager=self._context_manager,
            counter=self._counter,
            budget=self._budget,
            tool_result_max_tokens=self._settings.tool_result_max_tokens,
            settings=self._settings,
            registry=self._registry,
            provider=self._provider,
            engine=self._engine,
            turn_token=turn_token,
        )

    def _start_turn(self, conversation_id: int) -> _TurnRun:
        token = uuid.uuid4().hex
        return _TurnRun(
            conversation_id=conversation_id,
            token=token,
            started=time.perf_counter(),
            config=thread_config(conversation_id, turn_token=token),
        )

    async def _finish_turn(
        self, run, state, *, status, error_type=None, streaming=True
    ):
        if run is None or run.finished:
            return
        run.finished = True
        snapshot = {}
        if self._metrics_handler is not None:
            try:
                snapshot = self._metrics_handler.finish_turn(run.token) or {}
            except Exception:
                logger.warning("读取回调指标失败", exc_info=True)
        state = dict(state or {})
        citations = state.get("citations") or []
        payload = {
            "conversation_id": run.conversation_id,
            "turn_token": run.token,
            "trace_id": snapshot.get("trace_id"),
            "route": state.get("route"),
            "intent": state.get("intent"),
            "status": status,
            "error_type": (error_type or "")[:64] or None,
            "llm_calls": int(snapshot.get("llm_calls") or 0),
            "prompt_tokens": int(snapshot.get("prompt_tokens") or 0),
            "completion_tokens": int(snapshot.get("completion_tokens") or 0),
            "total_tokens": int(snapshot.get("total_tokens") or 0),
            "usage_estimated": bool(snapshot.get("usage_estimated")),
            "model_usage": snapshot.get("model_usage") or None,
            "stages": snapshot.get("stages") or None,
            "cost_cny": float(snapshot.get("cost_cny") or 0),
            "latency_ms": int((time.perf_counter() - run.started) * 1000),
            "first_token_ms": snapshot.get("first_token_ms") if streaming else None,
            "refusal_source": state.get("refusal_source"),
            "citations_count": len(citations),
        }
        if self._turn_metrics_writer is not None:
            await self._turn_metrics_writer.write(payload)

    def _turn_scope(
        self, conversation_id: int, token: str, *, message: str,
        user_id: str = "anonymous", mode: str = "chat",
    ):
        observer = (
            TurnObserver(self._metrics_handler, token)
            if self._metrics_handler is not None
            else None
        )
        trace = (
            self._trace_scope(
                conversation_id, message, user_id=user_id, mode=mode,
            )
            if self._trace_scope is not None
            else nullcontext()
        )
        return observer_scope(observer), trace

    @staticmethod
    def _finish_trace(root, state, *, payload=None, streamed=None):
        """把一轮的最终回答写进根 observation（trace 级 input/output 由它派生）。"""
        if root is None:
            return
        try:
            if payload is not None:
                actions = _interrupt_actions(payload)
                prompts = [a.get("prompt") for a in actions if a.get("prompt")]
                output = prompts[0] if prompts else "等待用户补充信息"
                status = "interrupted"
            else:
                output = (state or {}).get("answer") or streamed or ""
                status = "ok"
            root.update(
                output=[{"role": "assistant", "content": output}],
                metadata={
                    "route": (state or {}).get("route"),
                    "intent": (state or {}).get("intent"),
                    "status": status,
                },
            )
        except Exception:
            logger.debug("trace 输出写入失败", exc_info=True)

    async def shutdown(self) -> None:
        if self._summary_service is not None:
            await self._summary_service.shutdown()

    @staticmethod
    def _coerce_session_id(session_id: str | None) -> int | None:
        if session_id is None:
            return None
        try:
            return int(session_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("session_id 必须是数字") from exc

    async def _require_conversation(self, conversation_id: int) -> None:
        async with self._session_factory() as session:
            conv = await ChatRepository(session).get_conversation(conversation_id)
        if conv is None:
            raise ValueError("会话不存在")

    async def _reserve(self, conversation_id: int) -> _ResumeLock:
        """为校验阶段取会话锁并暂存，供后续流式生成器接管释放。"""
        lock = await _lock_for_conversation(conversation_id)
        await lock.acquire()
        guard = _ResumeLock(self._held_locks, conversation_id, lock)
        self._held_locks[conversation_id] = guard
        return guard

    def _release(self, conversation_id: int) -> None:
        guard = self._held_locks.pop(conversation_id, None)
        if guard is not None:
            guard.release_if_unclaimed()

    async def _reject_if_ticket_pending(self, conversation_id: int) -> None:
        """建单确认待处理时拒绝新聊天轮次，避免在 _prepare 落库用户消息前进入本轮。"""
        snapshot = await self._graph.aget_state(thread_config(conversation_id))
        payload = _interrupt_value(snapshot)
        if isinstance(payload, dict) and payload.get("type") == "ticket_confirmation":
            raise ValueError("当前会话有待确认的工单，请先确认或取消后再继续对话")

    async def pending_action_for(self, conversation_id: int) -> dict | None:
        """历史消息接口的可选 pending_action：仅工单确认卡片，无则 None。"""
        snapshot = await self._graph.aget_state(thread_config(conversation_id))
        payload = _interrupt_value(snapshot)
        for action in _interrupt_actions(payload):
            if action.get("type") == "ticket_preview":
                return action
        return None

    async def _prepare(self, session_id: str | None, message: str) -> _Prepared:
        if self._counter.count_text(message) > self._settings.max_user_input_tokens:
            raise ValueError(
                f"用户输入过长：超过 {self._settings.max_user_input_tokens} token 上限"
            )
        async with self._session_factory() as session:
            repo = ChatRepository(session)
            if session_id is None:
                conversation_id = await repo.create_conversation()
                await session.commit()
                created = True
                user_id = "anonymous"
            else:
                conversation_id = int(session_id)
                created = False
                conversation = await repo.get_conversation(conversation_id)
                if conversation is None:
                    raise ValueError("会话不存在")
                user_id = conversation.user_id or "anonymous"
            user_row = await repo.add_message(conversation_id, role="user", content=message)
            await repo.touch_conversation(conversation_id)
            await session.commit()
            new_user_id = user_row.id
            rows = await repo.list_messages(conversation_id)

        history_rows = [row for row in rows if row.id != new_user_id]
        config = thread_config(conversation_id)
        snapshot = await self._graph.aget_state(config)
        existing = list((snapshot.values or {}).get("messages") or []) if snapshot else []
        # ch08 建单意图证据：只从「上一轮 checkpoint 证据 + 本轮落库用户原话」推导，
        # 模型输出 / 工具参数 / create_ticket 参数一律无权注入。存 dict 以便 checkpoint 序列化。
        previous_ticket = (snapshot.values or {}).get("ticket_request") if snapshot else None
        ticket_evidence = update_ticket_request(previous_ticket, new_user_id, message)
        if existing:
            # checkpoint 里已有历史：本轮只发新消息，历史由 checkpointer 带给图。
            input_messages = []
            history_count = len(existing)
        else:
            # checkpoint 为空时，历史来自 MySQL 的补种消息（只种 user/assistant，
            # 且旧 assistant 的 tool_calls 一并丢弃，否则没有配对 ToolMessage 是协议错误）。
            input_messages = [
                seeded for row in history_rows
                if (seeded := _row_to_seed_message(row)) is not None
            ]
            history_count = len(input_messages)
        # 两种情况下都从这段历史之后开始算本轮：log_turn 用该偏移只统计本轮工具调用，
        # 落库也用它切出本轮新增消息，避免把历史重复写回 MySQL。
        turn_msg_offset = history_count
        input_messages.append(HumanMessage(content=message, id=f"db:{new_user_id}"))

        graph_input = {
            "messages": input_messages,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "turn_user_msg_id": new_user_id,
            "context_meta": {},
            "ticket_request": (
                ticket_evidence.to_dict() if ticket_evidence is not None else None
            ),
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
            "turn_msg_offset": turn_msg_offset,
            "tokens_used": 0,
            "suggested_actions": [],
            "resolved_query": None,
            "coref_source": None,
            "intent_confidence": None,
            "order_no": None,
            "order_data": None,
            "order_error": None,
            "policy_queries": [],
            "expand_degraded": None,
            "policy_refused": None,
            "refund_offered": None,
            "flow_steps": [],
            "fallback_reason": None,
            "stop_reason": None,
            "trace": {"t0": time.perf_counter()},
        }
        turn = TurnContext(
            raw_question=message,
            conversation_id=conversation_id,
            source_files=default_source_files(),
        )
        # 切片要跳过补种/已有历史，否则 /chat/agent 会把上一轮的工具调用重复算进本轮载荷。
        history_count = turn_msg_offset
        return _Prepared(
            conversation_id=conversation_id,
            created=created,
            turn=turn,
            graph_input=graph_input,
            history_count=history_count,
            user_id=user_id,
        )

    async def _persist_new(
        self, conversation_id: int, state: dict, history_count: int
    ) -> int | None:
        new_messages = list(state.get("messages") or [])[history_count:]
        last_message_id = None
        async with self._session_factory() as session:
            repo = ChatRepository(session)
            for message in new_messages:
                if not isinstance(message, AIMessage):
                    continue
                if str(getattr(message, "id", "") or "").startswith("db:"):
                    continue
                content = _text_of(message.content).strip()
                if not content:
                    continue
                row = await repo.add_message(
                    conversation_id, role="assistant", content=content,
                )
                last_message_id = row.id
            await repo.touch_conversation(conversation_id)
            await session.commit()
        return last_message_id

    async def _record_turn_snapshot(self, config, state, turn, message_id):
        if message_id is None:
            return
        citations = state.get("citations")
        if citations is None:            # only when the state key is missing (old checkpoint/exception)
            citations = turn.citations
        snapshots = dict(state.get("citations_by_message_id") or {})
        # 合并图里已持久化的快照：state 可能是本轮截断后的结果，不含历史快照；
        # 历史快照以 checkpointer 为准，否则每轮会丢上一轮之前的快照。
        try:
            snapshot = await self._graph.aget_state(config)
            existing = dict((snapshot.values or {}).get("citations_by_message_id") or {})
            existing.update(snapshots)
            snapshots = existing
        except Exception:
            pass
        snapshots[str(message_id)] = (
            None if citations is None else cap_citations(citations)
        )
        snapshots = dict(sorted(
            snapshots.items(), key=lambda kv: int(kv[0])
        )[-10:])
        try:
            await self._graph.aupdate_state(
                config,
                {"citations": citations, "citations_by_message_id": snapshots},
                as_node="log",
            )
        except Exception:
            logger.warning("citations 快照回写失败（不影响本轮回答）", exc_info=True)

    async def latest_turn_snapshot(self, conversation_id):
        async with self._session_factory() as session:
            latest_id = await ChatRepository(session).latest_assistant_message_id(
                conversation_id
            )
        if latest_id is None:
            return None, None
        try:
            snapshot = await self._graph.aget_state(thread_config(conversation_id))
            values = dict(snapshot.values or {}) if snapshot else {}
            snapshots = values.get("citations_by_message_id") or {}
            return latest_id, snapshots.get(str(latest_id))
        except Exception:
            logger.warning("读取 citations 快照失败，反馈留空", exc_info=True)
            return latest_id, None

    async def _events(self, graph_input, config, context):
        async for raw_event in self._graph.astream(
            graph_input,
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

    async def stream(self, session_id: str | None, message: str):
        try:
            conversation_id = self._coerce_session_id(session_id)
        except ValueError as exc:
            yield {"type": "error", "message": str(exc)}
            return
        lock = None
        if conversation_id is not None:
            lock = await _lock_for_conversation(conversation_id)
            await lock.acquire()
        run = None
        state = {}
        status = "interrupted"
        error_type = None
        try:
            if conversation_id is not None:
                await self._reject_if_ticket_pending(conversation_id)
            prepared = await self._prepare(session_id, message)
            if prepared.created:
                yield {"type": "session", "session_id": str(prepared.conversation_id)}
            run = self._start_turn(prepared.conversation_id)
            config = run.config
            context = self._context(prepared.turn, run.token)
            observer_cm, trace_cm = self._turn_scope(
                prepared.conversation_id, run.token, message=message,
                user_id=prepared.user_id, mode="stream",
            )
            with observer_cm, trace_cm as root:
                streamed: list[str] = []
                async for event in self._events(prepared.graph_input, config, context):
                    if event.get("type") == "token":
                        streamed.append(event.get("content") or "")
                    yield event
                snapshot = await self._graph.aget_state(config)
                state = dict(snapshot.values or {}) if snapshot else {}
                payload = _interrupt_value(snapshot)
                self._finish_trace(
                    root, state, payload=payload, streamed="".join(streamed),
                )
            if payload is not None:
                actions = _interrupt_actions(payload)
                kinds = [a.get("type") for a in actions]
                logger.info(json.dumps({
                    "conversation_id": prepared.conversation_id,
                    "stop_reason": "awaiting_ticket" if "ticket_preview" in kinds else "awaiting_order",
                    "orders_count": len((payload or {}).get("orders") or []),
                }, ensure_ascii=False))
                if actions:
                    yield {"type": "actions", "items": actions}
                status = "ok"
                yield {"type": "done"}
                return

            message_id = await self._persist_new(
                prepared.conversation_id, state, prepared.history_count
            )
            await self._record_turn_snapshot(config, state, prepared.turn, message_id)
            citations = state.get("citations") or prepared.turn.citations
            if citations:
                yield {"type": "citations", "items": citations}
            actions = state.get("suggested_actions") or []
            if actions:
                yield {"type": "actions", "items": actions}
            status = "ok"
            done = {"type": "done"}
            if message_id is not None:
                done["message_id"] = message_id
            yield done
        except GeneratorExit:
            status = "cancelled"
            raise
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            yield {"type": "error", "message": str(exc)}
        finally:
            if run is not None:
                await self._finish_turn(
                    run, state, status=status, error_type=error_type, streaming=True
                )
            if lock is not None:
                lock.release()

    async def run(self, session_id: str | None, message: str) -> dict:
        conversation_id = self._coerce_session_id(session_id)
        lock = None
        if conversation_id is not None:
            lock = await _lock_for_conversation(conversation_id)
            await lock.acquire()
        run = None
        state = {}
        status = "interrupted"
        error_type = None
        try:
            if conversation_id is not None:
                await self._reject_if_ticket_pending(conversation_id)
            prepared = await self._prepare(session_id, message)
            run = self._start_turn(prepared.conversation_id)
            config = run.config
            context = self._context(prepared.turn, run.token)
            observer_cm, trace_cm = self._turn_scope(
                prepared.conversation_id, run.token, message=message,
                user_id=prepared.user_id, mode="agent",
            )
            with observer_cm, trace_cm as root:
                state = await self._graph.ainvoke(
                    prepared.graph_input,
                    config=config,
                    context=context,
                )
                state = dict(state or {})
                payload = _interrupt_from_result(state)
                if payload is None:
                    snapshot = await self._graph.aget_state(config)
                    payload = _interrupt_value(snapshot)
                self._finish_trace(root, state, payload=payload)
            if payload is not None:
                status = "ok"
                return {
                    "session_id": prepared.conversation_id,
                    "interrupted": True,
                    "actions": _interrupt_actions(payload),
                    "trace": state.get("trace") or {},
                }
            message_id = await self._persist_new(
                prepared.conversation_id, state, prepared.history_count
            )
            await self._record_turn_snapshot(config, state, prepared.turn, message_id)
            result = self._run_result(
                prepared.conversation_id, state, prepared.history_count
            )
            result["interrupted"] = False
            result["message_id"] = message_id
            status = "ok"
            return result
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            if run is not None:
                await self._finish_turn(
                    run, state, status=status, error_type=error_type, streaming=False
                )
            if lock is not None:
                lock.release()

    def _run_result(self, conversation_id, state, history_count):
        messages = list(state.get("messages") or [])
        new_messages = messages[history_count:]
        tool_calls = []
        tool_results = []
        for item in new_messages:
            if isinstance(item, AIMessage):
                tool_calls.extend(item.tool_calls or [])
            elif isinstance(item, ToolMessage):
                tool_results.append(
                    {"tool_call_id": item.tool_call_id, "content": _text_of(item.content)}
                )
        actions = state.get("suggested_actions") or []
        return {
            "session_id": conversation_id,
            "answer": state.get("answer"),
            "intent": state.get("intent"),
            "route": state.get("route"),
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "suggested_actions": actions,
            "actions": actions,
            "trace": state.get("trace") or {},
        }

    async def _resume_setup(self, session_id: str, order_no: str):
        conversation_id = self._coerce_session_id(session_id)
        async with self._session_factory() as session:
            repo = ChatRepository(session)
            conv = await repo.get_conversation(conversation_id)
            if conv is None:
                raise ValueError("会话不存在")
            order = await OrderRepository(session).get(order_no)
            if order is None or order.user_id != conv.user_id:
                raise ValueError("订单不存在")
        config = thread_config(conversation_id)
        snapshot = await self._graph.aget_state(config)
        payload = _interrupt_value(snapshot)
        if not isinstance(payload, dict) or payload.get("kind") != "select_order":
            raise ValueError("会话没有待选择的订单")
        messages = list((snapshot.values or {}).get("messages") or [])
        turn = TurnContext(
            raw_question=_last_human_text(messages),
            conversation_id=conversation_id,
            source_files=default_source_files(),
        )
        return conversation_id, config, snapshot, len(messages), turn

    async def resume_stream(self, session_id: str, order_no: str):
        conversation_id = self._coerce_session_id(session_id)
        guard = self._held_locks.pop(conversation_id, None)
        if guard is not None:
            lock = guard.claim()
        else:
            lock = await _lock_for_conversation(conversation_id)
            await lock.acquire()
        # 校验阶段的 ValueError（会话/订单不存在）必须向调用方传播，与 validate_resume 的
        # 拒绝口径一致；graph 执行阶段的异常才转成 error 事件。锁在这里由本分支兜底释放。
        try:
            conversation_id, _, snapshot, history_count, turn = await self._resume_setup(
                session_id, order_no
            )
        except BaseException:
            lock.release()
            raise
        run = None
        state = {}
        status = "interrupted"
        error_type = None
        try:
            run = self._start_turn(conversation_id)
            config = run.config
            context = self._context(turn, run.token)
            observer_cm, trace_cm = self._turn_scope(
                conversation_id, run.token, message=f"选择订单 {order_no}",
                mode="stream-resume",
            )
            with observer_cm, trace_cm as root:
                streamed: list[str] = []
                async for event in self._events(
                    Command(resume={"order_no": order_no}), config, context
                ):
                    if event.get("type") == "token":
                        streamed.append(event.get("content") or "")
                    yield event
                snapshot = await self._graph.aget_state(config)
                state = dict(snapshot.values or {}) if snapshot else {}
                payload = _interrupt_value(snapshot)
                self._finish_trace(
                    root, state, payload=payload, streamed="".join(streamed),
                )
            if payload is not None:
                for action in _interrupt_actions(payload):
                    yield {"type": "actions", "items": [action]}
                status = "ok"
                yield {"type": "done"}
                return
            message_id = await self._persist_new(conversation_id, state, history_count)
            await self._record_turn_snapshot(config, state, turn, message_id)
            citations = state.get("citations") or turn.citations
            if citations:
                yield {"type": "citations", "items": citations}
            actions = state.get("suggested_actions") or []
            if actions:
                yield {"type": "actions", "items": actions}
            status = "ok"
            done = {"type": "done"}
            if message_id is not None:
                done["message_id"] = message_id
            yield done
        except GeneratorExit:
            status = "cancelled"
            raise
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            yield {"type": "error", "message": str(exc)}
        finally:
            if run is not None:
                await self._finish_turn(
                    run, state, status=status, error_type=error_type, streaming=True
                )
            lock.release()

    async def validate_resume(self, session_id: str, order_no: str) -> _ResumeLock:
        """Validate before opening the SSE stream so /chat/resume can return 400/404.

        校验成功后返回已持有的会话锁守卫；路由把守卫传给 SSE 响应，在流未接管时兜底释放。
        """
        conversation_id = self._coerce_session_id(session_id)
        await self._require_conversation(conversation_id)
        guard = await self._reserve(conversation_id)
        try:
            await self._resume_setup(session_id, order_no)
        except BaseException:
            self._release(conversation_id)
            raise
        return guard

    async def run_resume(self, session_id: str, order_no: str) -> dict:
        conversation_id = self._coerce_session_id(session_id)
        lock = await _lock_for_conversation(conversation_id)
        await lock.acquire()
        run = None
        state = {}
        status = "interrupted"
        error_type = None
        try:
            conversation_id, _, snapshot, history_count, turn = await self._resume_setup(
                session_id, order_no
            )
            run = self._start_turn(conversation_id)
            config = run.config
            context = self._context(turn, run.token)
            observer_cm, trace_cm = self._turn_scope(
                conversation_id, run.token, message=f"选择订单 {order_no}",
                mode="agent-resume",
            )
            with observer_cm, trace_cm as root:
                state = await self._graph.ainvoke(
                    Command(resume={"order_no": order_no}),
                    config=config,
                    context=context,
                )
                state = dict(state or {})
                payload = _interrupt_from_result(state)
                if payload is None:
                    snapshot = await self._graph.aget_state(config)
                    payload = _interrupt_value(snapshot)
                self._finish_trace(root, state, payload=payload)
            if payload is not None:
                status = "ok"
                return {
                    "session_id": conversation_id,
                    "interrupted": True,
                    "actions": _interrupt_actions(payload),
                    "trace": state.get("trace") or {},
                }
            message_id = await self._persist_new(conversation_id, state, history_count)
            await self._record_turn_snapshot(config, state, turn, message_id)
            result = self._run_result(conversation_id, state, history_count)
            result["interrupted"] = False
            result["message_id"] = message_id
            status = "ok"
            return result
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            if run is not None:
                await self._finish_turn(
                    run, state, status=status, error_type=error_type, streaming=False
                )
            lock.release()

    async def _validate_ticket_confirmation(self, conversation_id: int, confirmation_id: str) -> None:
        """校验已取锁后的工单恢复请求；失败抛 TicketResumeRejected。"""
        snapshot = await self._graph.aget_state(thread_config(conversation_id))
        payload = _interrupt_value(snapshot)
        if not isinstance(payload, dict) or payload.get("type") != "ticket_confirmation":
            raise TicketResumeRejected(409, "会话没有待确认的工单")
        draft = payload.get("draft") or {}
        if int(draft.get("conversation_id") or 0) != conversation_id:
            raise TicketResumeRejected(409, "工单草稿与会话不一致")
        if payload.get("confirmation_id") != confirmation_id:
            raise TicketResumeRejected(409, "确认标识已失效，请重新加载")
        evidence = (snapshot.values or {}).get("ticket_request")
        if getattr(evidence, "to_dict", None):
            evidence = evidence.to_dict()
        if (not isinstance(evidence, dict) or evidence.get("status") != "active"
                or int(evidence.get("message_id") or 0) != int(draft.get("request_message_id") or 0)):
            raise TicketResumeRejected(409, "建单请求证据已失效，请重新发起")
        from app.tools.builtin import build_tool_snapshot

        definitions = await build_tool_snapshot(self._registry, self._provider)
        definition = definitions.get("create_ticket")
        if definition is None:
            raise TicketResumeRejected(403, "create_ticket 未注册，无法建单")
        if not self._engine.policy.load().permits(definition):
            raise TicketResumeRejected(403, "create_ticket 权限被拒绝")

    async def validate_ticket_resume(self, session_id: str, confirmation_id: str, decision: str) -> _ResumeLock:
        try:
            conversation_id = int(session_id)
        except (TypeError, ValueError):
            raise TicketResumeRejected(400, "session_id 必须是数字")
        async with self._session_factory() as session:
            conv = await ChatRepository(session).get_conversation(conversation_id)
        if conv is None:
            raise TicketResumeRejected(404, "会话不存在")
        guard = await self._reserve(conversation_id)
        try:
            await self._validate_ticket_confirmation(conversation_id, confirmation_id)
        except BaseException:
            self._release(conversation_id)
            raise
        return guard

    async def ticket_resume_stream(self, session_id: str, confirmation_id: str, decision: str):
        conversation_id = int(session_id)
        guard = self._held_locks.pop(conversation_id, None)
        if guard is not None:
            lock = guard.claim()
        else:
            lock = await _lock_for_conversation(conversation_id)
            await lock.acquire()
        run = None
        state = {}
        status = "interrupted"
        error_type = None
        try:
            run = self._start_turn(conversation_id)
            config = run.config
            snapshot = await self._graph.aget_state(config)
            messages = list((snapshot.values or {}).get("messages") or [])
            turn = TurnContext(
                raw_question=_last_human_text(messages),
                conversation_id=conversation_id,
                source_files=default_source_files(),
            )
            history_count = len(messages)
            context = self._context(turn, run.token)
            observer_cm, trace_cm = self._turn_scope(
                conversation_id, run.token,
                message=f"工单确认 {decision} {confirmation_id}",
                mode="stream-ticket-resume",
            )
            with observer_cm, trace_cm as root:
                streamed: list[str] = []
                async for event in self._events(
                    Command(resume={"decision": decision, "confirmation_id": confirmation_id}),
                    config, context,
                ):
                    if event.get("type") == "token":
                        streamed.append(event.get("content") or "")
                    yield event
                snapshot = await self._graph.aget_state(config)
                state = dict(snapshot.values or {}) if snapshot else {}
                payload = _interrupt_value(snapshot)
                self._finish_trace(
                    root, state, payload=payload, streamed="".join(streamed),
                )
            if payload is not None:
                for action in _interrupt_actions(payload):
                    yield {"type": "actions", "items": [action]}
                status = "ok"
                yield {"type": "done"}
                return
            message_id = await self._persist_new(conversation_id, state, history_count)
            await self._record_turn_snapshot(config, state, turn, message_id)
            citations = state.get("citations") or turn.citations
            if citations:
                yield {"type": "citations", "items": citations}
            actions = state.get("suggested_actions") or []
            if actions:
                yield {"type": "actions", "items": actions}
            status = "ok"
            done = {"type": "done"}
            if message_id is not None:
                done["message_id"] = message_id
            yield done
        except GeneratorExit:
            status = "cancelled"
            raise
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            yield {"type": "error", "message": str(exc)}
        finally:
            if run is not None:
                await self._finish_turn(
                    run, state, status=status, error_type=error_type, streaming=True
                )
            lock.release()

    async def create_ticket_button(self, session_id: str, ticket_type: str, description: str) -> str:
        """旧 /chat/ticket 按钮：服务端自持草稿与确认，仍走同一引擎并留审计，不自动升级。"""
        from app.tools.builtin import build_tool_snapshot

        conversation_id = self._coerce_session_id(session_id)
        async with self._session_factory() as session:
            conv = await ChatRepository(session).get_conversation(conversation_id)
            if conv is None:
                raise ValueError("会话不存在")
            rows = await ChatRepository(session).list_messages(conversation_id)
        user_rows = [row for row in rows if row.role.value == "user"]
        if not user_rows:
            raise ValueError("会话没有用户消息，无法建单")
        newest = user_rows[-1]
        evidence = {
            "message_id": newest.id,
            "quote": (newest.content or "")[:200],
            "status": "active",
        }
        tool_call_id = f"btn-{uuid.uuid4().hex}"
        args = {"ticket_type": ticket_type, "description": description}
        context = ExecutionContext(
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            session_factory=self._session_factory,
            ticket_request=evidence,
            confirmation=_auto_approve_confirmation,
            state={"route": "business", "intent": None},
        )
        context.escalate = False
        definitions = await build_tool_snapshot(self._registry, self._provider)
        message = await self._engine.execute(
            {"name": "create_ticket", "id": tool_call_id, "args": args}, context, definitions,
        )
        if (getattr(message, "artifact", None) or {}).get("category") != "success":
            raise TicketEngineFault("工单创建失败，请稍后重试")
        return ticket_no_for(draft_from_call(args, context))


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


def build_chat_service(
    graph, *, settings=None, session_factory, registry=None, provider=None, engine=None,
    metrics_handler=None, turn_metrics_writer=None, trace_scope=None,
):
    from app.core.llm import build_chat_model, build_structured_model
    from app.prompts.context import SummaryOutput
    from app.prompts.orchestrator import CorefOutput, ExpansionOutput, IntentOutput

    s = settings or default_settings
    counter = TokenCounter(
        cjk_tokens_per_char=s.cjk_tokens_per_char,
        estimate_factor=s.token_estimate_factor,
    )
    budget = compute_budget(s)
    summary_service = SummaryService(
        session_factory,
        build_structured_model(
            s, SummaryOutput, temperature=0.0,
            max_tokens=s.summary_max_output_tokens,
        ),
        graph,
        counter,
        s,
    )
    manager = ContextManager(
        session_factory,
        counter=counter,
        budget=budget,
        settings=s,
        summary_service=summary_service,
    )
    return ChatService(
        build_chat_model(s, max_tokens=s.max_output_tokens),
        build_structured_model(
            s, IntentOutput, temperature=0.0, model_name=s.intent_model or None,
        ),
        session_factory,
        graph,
        token_budget=s.token_budget,
        tool_timeout=s.tool_timeout,
        tool_retries=s.tool_retries,
        coref_model=build_structured_model(s, CorefOutput, temperature=0.0),
        expansion_model=build_structured_model(s, ExpansionOutput, temperature=0.0),
        intent_mode=s.intent_mode,
        settings=s,
        context_manager=manager,
        summary_service=summary_service,
        counter=counter,
        registry=registry,
        provider=provider,
        engine=engine,
        metrics_handler=metrics_handler,
        turn_metrics_writer=turn_metrics_writer,
        trace_scope=trace_scope,
    )
