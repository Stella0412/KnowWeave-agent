"""图驱动的聊天服务：准备会话 → 跑 LangGraph → 映射 SSE → 增量落 MySQL。"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

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
from app.rag.pipeline import TurnContext, default_source_files
from app.services.context_manager import ContextManager
from app.services.summary_service import SummaryService
from app.services.token_counter import TokenCounter, compute_budget
from app.tools.ticket_authorization import update_ticket_request

logger = logging.getLogger("knowweave.turn")

_FIXED_REPLY_NODES = {"chitchat_reply", "complaint_reply", "fallback_reply"}


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
    if not isinstance(payload, dict) or payload.get("kind") != "select_order":
        return []
    return [{
        "type": "select_order",
        "label": "选择订单",
        "prompt": payload.get("prompt") or "请选择要退款的订单",
        "orders": list(payload.get("orders") or []),
    }]


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

    def _context(self, turn: TurnContext) -> GraphContext:
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
        )

    async def shutdown(self) -> None:
        if self._summary_service is not None:
            await self._summary_service.shutdown()

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
            else:
                conversation_id = int(session_id)
                created = False
                if await repo.get_conversation(conversation_id) is None:
                    raise ValueError("会话不存在")
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
            "user_id": "anonymous",
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
        )

    async def _persist_new(self, conversation_id: int, state: dict, history_count: int) -> None:
        new_messages = list(state.get("messages") or [])[history_count:]
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
                await repo.add_message(
                    conversation_id, role="assistant", content=content,
                )
            await repo.touch_conversation(conversation_id)
            await session.commit()

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
            prepared = await self._prepare(session_id, message)
            if prepared.created:
                yield {"type": "session", "session_id": str(prepared.conversation_id)}
            config = thread_config(prepared.conversation_id)
            context = self._context(prepared.turn)
            async for event in self._events(prepared.graph_input, config, context):
                yield event

            snapshot = await self._graph.aget_state(config)
            payload = _interrupt_value(snapshot)
            if payload is not None:
                actions = _interrupt_actions(payload)
                logger.info(json.dumps({
                    "conversation_id": prepared.conversation_id,
                    "stop_reason": "awaiting_order",
                    "orders_count": len((payload or {}).get("orders") or []),
                }, ensure_ascii=False))
                if actions:
                    yield {"type": "actions", "items": actions}
                yield {"type": "done"}
                return

            state = dict(snapshot.values or {}) if snapshot else {}
            await self._persist_new(prepared.conversation_id, state, prepared.history_count)
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
        payload = _interrupt_from_result(state)
        if payload is None:
            snapshot = await self._graph.aget_state(config)
            payload = _interrupt_value(snapshot)
        if payload is not None:
            return {
                "session_id": prepared.conversation_id,
                "interrupted": True,
                "actions": _interrupt_actions(payload),
                "trace": state.get("trace") or {},
            }
        await self._persist_new(prepared.conversation_id, state, prepared.history_count)
        result = self._run_result(prepared.conversation_id, state, prepared.history_count)
        result["interrupted"] = False
        return result

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
        try:
            conversation_id = int(session_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("session_id 必须是数字") from exc
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
        if _interrupt_value(snapshot) is None:
            raise ValueError("会话没有待选择的订单")
        messages = list((snapshot.values or {}).get("messages") or [])
        turn = TurnContext(
            raw_question=_last_human_text(messages),
            conversation_id=conversation_id,
            source_files=default_source_files(),
        )
        return conversation_id, config, snapshot, len(messages), turn

    async def resume_stream(self, session_id: str, order_no: str):
        conversation_id, config, snapshot, history_count, turn = await self._resume_setup(
            session_id, order_no
        )
        context = self._context(turn)
        async for event in self._events(
            Command(resume={"order_no": order_no}), config, context
        ):
            yield event
        snapshot = await self._graph.aget_state(config)
        payload = _interrupt_value(snapshot)
        if payload is not None:
            for action in _interrupt_actions(payload):
                yield {"type": "actions", "items": [action]}
            yield {"type": "done"}
            return
        state = dict(snapshot.values or {}) if snapshot else {}
        await self._persist_new(conversation_id, state, history_count)
        citations = state.get("citations") or turn.citations
        if citations:
            yield {"type": "citations", "items": citations}
        actions = state.get("suggested_actions") or []
        if actions:
            yield {"type": "actions", "items": actions}
        yield {"type": "done"}

    async def validate_resume(self, session_id: str, order_no: str) -> None:
        """Validate before opening the SSE stream so /chat/resume can return 400/404."""
        await self._resume_setup(session_id, order_no)

    async def run_resume(self, session_id: str, order_no: str) -> dict:
        conversation_id, config, snapshot, history_count, turn = await self._resume_setup(
            session_id, order_no
        )
        state = await self._graph.ainvoke(
            Command(resume={"order_no": order_no}),
            config=config,
            context=self._context(turn),
        )
        state = dict(state or {})
        payload = _interrupt_from_result(state)
        if payload is None:
            snapshot = await self._graph.aget_state(config)
            payload = _interrupt_value(snapshot)
        if payload is not None:
            return {
                "session_id": conversation_id,
                "interrupted": True,
                "actions": _interrupt_actions(payload),
                "trace": state.get("trace") or {},
            }
        await self._persist_new(conversation_id, state, history_count)
        result = self._run_result(conversation_id, state, history_count)
        result["interrupted"] = False
        return result


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
    graph, *, settings=None, session_factory, registry=None, provider=None, engine=None
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
    )
