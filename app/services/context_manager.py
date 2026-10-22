"""会话上下文的分层规划器。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.db.repository import ChatRepository
from app.prompts.customer_service import SYSTEM_PROMPT
from app.services.token_counter import TokenCounter, compute_budget, text_of
from app.tools.registry import build_tools

logger = logging.getLogger("knowweave.context")


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


class ContextManager:
    def __init__(
        self,
        session_factory,
        *,
        counter: TokenCounter,
        budget,
        settings,
        summary_service=None,
    ):
        self._session_factory = session_factory
        self._counter = counter
        self._budget = budget
        self._settings = settings
        self._summary_service = summary_service

    def _flatten(self, turns: list[Turn]) -> list[BaseMessage]:
        return [message for turn in turns for message in turn.messages]

    def _partition(self, messages, summary_upto, layer1_from, current_id):
        turns = split_turns(messages)
        current = None
        history: list[Turn] = []
        for turn in turns:
            if current_id is not None and turn.first_user_msg_id == current_id:
                current = turn
                continue
            history.append(turn)
        layer3 = [t for t in history if classify_turn(t, summary_upto, layer1_from) == 3]
        layer2 = [t for t in history if classify_turn(t, summary_upto, layer1_from) == 2]
        layer1 = [t for t in history if classify_turn(t, summary_upto, layer1_from) == 1]
        return turns, current, layer3, layer2, layer1

    async def prepare(self, conversation_id: int, messages, current_user_msg_id: int) -> dict:
        async with self._session_factory() as session:
            repo = ChatRepository(session)
            conv = await repo.get_conversation(conversation_id)
            if conv is None:
                raise ValueError("会话不存在")
            segments = await repo.list_summaries(conversation_id)

        summary_upto = conv.summary_upto_msg_id
        max_upto = max((row.upto_msg_id for row in segments), default=None)
        if summary_upto is None and max_upto is not None:
            logger.warning(json.dumps({
                "event": "anchor_normalized", "conversation_id": conversation_id,
                "summary_upto": max_upto,
            }, ensure_ascii=False))
            summary_upto = max_upto

        layer1_from = conv.layer1_from_msg_id
        stored_layer1_from = layer1_from
        turns = split_turns(messages)
        anchored = [t.first_user_msg_id for t in turns if t.first_user_msg_id is not None]
        if layer1_from is None:
            after = [anchor for anchor in anchored if summary_upto is None or anchor > summary_upto]
            layer1_from = after[0] if after else (anchored[0] if anchored else None)

        effective_summary = summary_upto
        if summary_upto is not None and layer1_from is not None and summary_upto >= layer1_from:
            valid = [row.upto_msg_id for row in segments if row.upto_msg_id < layer1_from]
            effective_summary = max(valid) if valid else None
            logger.warning(json.dumps({
                "event": "anchor_inconsistent", "conversation_id": conversation_id,
                "summary_upto": summary_upto, "layer1_from": layer1_from,
                "effective_summary_upto": effective_summary,
            }, ensure_ascii=False))

        _, current, _, layer2, layer1 = self._partition(
            messages, effective_summary, layer1_from, current_user_msg_id
        )
        layer1_tokens = self._counter.count_messages(self._flatten(layer1))
        demoted_from = layer1_from
        demoted_count = 0
        while layer1_tokens > self._budget.layer1 and len(layer1) > 1:
            moved = layer1.pop(0)
            layer2.insert(0, moved)
            demoted_count += 1
            layer1_tokens = self._counter.count_messages(self._flatten(layer1))
        if layer1_tokens > self._budget.layer1:
            logger.warning(json.dumps({
                "event": "layer1_overflow", "conversation_id": conversation_id,
                "tokens": layer1_tokens, "budget": self._budget.layer1,
            }, ensure_ascii=False))

        new_layer1_from = (
            layer1[0].first_user_msg_id if layer1 else current_user_msg_id
        )
        # 用库里的原始值比较：R12 首次把 NULL 初始化成实际锚点也要落库（无降级日志），
        # 不能和已在内存中更新过的 layer1_from 比较，否则 NULL 会一直拖到首次降级才写。
        if stored_layer1_from is None or new_layer1_from != stored_layer1_from:
            async with self._session_factory() as session:
                repo = ChatRepository(session)
                row = await repo.get_conversation_for_update(conversation_id)
                if row is not None and (
                    row.layer1_from_msg_id is None or new_layer1_from > row.layer1_from_msg_id
                ):
                    row.layer1_from_msg_id = new_layer1_from
                    await session.commit()
        # 首次把 NULL 落成实际值不是降级；只有真的整轮移动才允许打降级日志。
        if demoted_count:
            logger.info(json.dumps({
                "event": "layer1 降级", "conversation_id": conversation_id,
                "from": demoted_from, "to": new_layer1_from,
                "batch": demoted_count,
            }, ensure_ascii=False))
        layer1_from = new_layer1_from

        layer2_raw = self._counter.count_messages(self._flatten(layer2))
        summary_triggered = False
        if (
            layer2_raw > self._budget.layer2
            and self._summary_service is not None
            and layer2
        ):
            frozen = self._freeze(layer2, effective_summary)
            if frozen is not None:
                from app.services.summary_service import SummaryRange

                summary_triggered = await self._summary_service.schedule(
                    conversation_id,
                    SummaryRange(
                        from_msg_id=frozen[0].first_user_msg_id,
                        upto_msg_id=frozen[-1].first_user_msg_id,
                        expected_summary_upto=effective_summary,
                    ),
                )
                logger.info(json.dumps({
                    "event": "summary trigger", "conversation_id": conversation_id,
                    "layer2_tokens": layer2_raw, "budget": self._budget.layer2,
                    "from_msg_id": frozen[0].first_user_msg_id,
                    "upto_msg_id": frozen[-1].first_user_msg_id,
                }, ensure_ascii=False))

        summary_text, selected = self.select_summary(
            segments, self._settings.summary_inject_max_tokens
        )
        meta = {
            "conversation_id": conversation_id,
            "turn_user_msg_id": current_user_msg_id,
            "summary_upto": summary_upto,
            "effective_summary_upto": effective_summary,
            "layer1_from": layer1_from,
            "summary_text": summary_text,
            "summary_segments": selected,
            "layer1_tokens": layer1_tokens,
            "layer2_tokens": layer2_raw,
            "layer1_budget": self._budget.layer1,
            "layer2_budget": self._budget.layer2,
            "demoted": demoted_count > 0,
            "summary_triggered": summary_triggered,
            "budget": self._budget.as_dict(),
        }
        window = self.shared_history(messages, meta)
        logger.info(json.dumps({
            "event": "history_ctx", "conversation_id": conversation_id,
            "turn_user_msg_id": current_user_msg_id,
            "summary": summary_text, "summary_segments": selected,
            "window": [serialize_message(m) for m in window],
            "window_count": len(window),
            "estimated_tokens": self._counter.count_messages(window),
            "budget": self._settings.classifier_history_max_tokens,
            "anchors": {
                "summary_upto": summary_upto, "layer1_from": layer1_from,
            },
        }, ensure_ascii=False))
        return meta

    def _freeze(self, layer2: list[Turn], summary_upto):
        selected: list[Turn] = []
        total = 0
        for turn in layer2:
            tokens = self._counter.count_messages(turn.messages)
            if selected and total + tokens > self._settings.summary_input_max_tokens:
                break
            selected.append(turn)
            total += tokens
        return selected or None

    def select_summary(self, segments, max_tokens: int) -> tuple[str, list[int]]:
        if not segments:
            return "", []
        rendered = [
            (row.seq, f"【第{row.seq}段】{row.content}") for row in segments
        ]
        full = "\n".join(text for _, text in rendered)
        if self._counter.count_text(full) <= max_tokens:
            return full, [seq for seq, _ in rendered]
        first_seq, first_text = rendered[0]
        kept = [first_text]
        used = self._counter.count_text(first_text)
        recent: list[tuple[int, str]] = []
        for seq, text in reversed(rendered[1:]):
            cost = self._counter.count_text(text)
            if used + cost > max_tokens:
                continue
            recent.append((seq, text))
            used += cost
        recent.reverse()
        if recent:
            omitted = len(rendered) - 1 - len(recent)
            middle = [f"【中间省略 {omitted} 段】"] if omitted else []
            parts = kept + middle + [text for _, text in recent]
            selected = [first_seq] + [seq for seq, _ in recent]
        else:
            parts = [self._counter.truncate_text(first_text, max_tokens, "…（摘要已截断）")]
            selected = [first_seq]
        return "\n".join(parts), selected

    def model_history(self, messages, meta: dict):
        prefix, current, react = [], None, []
        if meta:
            turns = split_turns(messages)
            current_turn = next(
                (
                    turn for turn in turns
                    if turn.first_user_msg_id == meta.get("turn_user_msg_id")
                ),
                None,
            )
            layer2_turns: list[Turn] = []
            layer1_messages: list[BaseMessage] = []
            for turn in turns:
                if current_turn is not None and turn is current_turn:
                    current = turn.messages[0]
                    # 当前轮的占位 AI（无内容、无工具调用）不代表任何信息，不属于 ReAct 消息。
                    react = [
                        message
                        for message in turn.messages[1:]
                        if not (
                            isinstance(message, AIMessage)
                            and not text_of(message.content)
                            and not message.tool_calls
                        )
                    ]
                    continue
                if current_turn is not None and turn.first_user_msg_id == current_turn.first_user_msg_id:
                    continue
                layer = classify_turn(
                    turn, meta.get("effective_summary_upto"), meta.get("layer1_from")
                )
                if layer == 2:
                    layer2_turns.append(turn)
                elif layer == 1:
                    layer1_messages.extend(turn.messages)
            layer2_messages = [
                message
                for turn in layer2_turns
                for message in render_layer2_turn(
                    turn, self._settings.layer2_assistant_prefix_chars
                )
            ]
            if self._counter.count_messages(layer2_messages) > self._budget.layer2:
                kept_turns: list[Turn] = []
                kept_messages: list[BaseMessage] = []
                for turn in reversed(layer2_turns):
                    rendered = render_layer2_turn(
                        turn, self._settings.layer2_assistant_prefix_chars
                    )
                    candidate = rendered + kept_messages
                    if kept_messages and (
                        self._counter.count_messages(candidate) > self._budget.layer2
                    ):
                        break
                    kept_turns.insert(0, turn)
                    kept_messages = candidate
                dropped = len(layer2_turns) - len(kept_turns)
                if dropped > 0:
                    layer2_messages = [
                        HumanMessage(content="【层 2 早期内容待摘要】")
                    ] + kept_messages
                    logger.warning(json.dumps({
                        "event": "layer2_overflow",
                        "conversation_id": meta.get("conversation_id"),
                        "tokens": self._counter.count_messages(layer2_messages),
                        "budget": self._budget.layer2,
                        "dropped_turns": dropped,
                    }, ensure_ascii=False))
                else:
                    layer2_messages = kept_messages
            prefix = layer2_messages + layer1_messages
        if current is None:
            turns = split_turns(messages)
            if turns:
                current_turn = turns[-1]
                prefix = self._flatten(turns[:-1])
                current = current_turn.messages[0] if current_turn.messages else None
                react = [
                    message
                    for message in current_turn.messages[1:]
                    if not (
                        isinstance(message, AIMessage)
                        and not text_of(message.content)
                        and not message.tool_calls
                    )
                ]
        return prefix, current, react

    def shared_history(self, messages, meta: dict):
        prefix, _, _ = self.model_history(messages, meta)
        trimmed = trim_messages(
            prefix,
            max_tokens=self._settings.classifier_history_max_tokens,
            token_counter=self._counter,
            strategy="last",
            allow_partial=False,
            start_on="human",
            include_system=False,
        )
        while trimmed and isinstance(trimmed[0], ToolMessage):
            trimmed.pop(0)
        if not trimmed and prefix:
            # 预算小于单条消息时 trim 会清空；至少保留最新一条用户消息，
            # 否则分类器/指代消解没有上下文可用。
            last_human = next(
                (m for m in reversed(prefix) if isinstance(m, HumanMessage)), None
            )
            fallback = last_human if last_human is not None else prefix[-1]
            trimmed = [self._cap_message(fallback)]
        return trimmed

    def _cap_message(self, message: BaseMessage) -> BaseMessage:
        """把单条消息截到 classifier_history_max_tokens 预算内（含协议开销）。"""
        max_tokens = self._settings.classifier_history_max_tokens
        if self._counter.count_message(message) <= max_tokens:
            return message
        text = text_of(getattr(message, "content", ""))
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = message.model_copy(update={"content": text[:mid]})
            if self._counter.count_message(candidate) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        return message.model_copy(update={"content": text[:lo]})


def measure_system_prefix(counter: TokenCounter, session_factory=None) -> int:
    tools = build_tools(session_factory, 0)
    schemas = [convert_to_openai_tool(tool) for tool in tools]
    schema_text = json.dumps(schemas, ensure_ascii=False, sort_keys=True)
    return counter.count_text(SYSTEM_PROMPT) + counter.count_text(schema_text)


def log_startup_budget(logger, settings, counter: TokenCounter):
    budget = compute_budget(settings)
    prefix_tokens = measure_system_prefix(counter, session_factory=None)
    if prefix_tokens > settings.system_prefix_reserve_tokens:
        raise RuntimeError(
            f"系统前缀超出预算：{prefix_tokens} > {settings.system_prefix_reserve_tokens}"
        )
    logger.info(json.dumps({
        "event": "context_budget",
        **budget.as_dict(),
        "system_prefix_tokens": prefix_tokens,
        "components": {
            "system_prefix_reserve": settings.system_prefix_reserve_tokens,
            "evidence": settings.rerank_top_n * settings.evidence_tokens_per_chunk,
            "summary_inject": settings.summary_inject_max_tokens,
            "output": settings.max_output_tokens,
            "user_input": settings.max_user_input_tokens,
            "safety": settings.context_safety_margin_tokens,
        },
    }, ensure_ascii=False))
    return budget
