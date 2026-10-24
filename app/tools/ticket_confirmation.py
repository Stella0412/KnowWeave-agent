"""Server-owned ticket drafts, LangGraph confirmation, and idempotent writes."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import asdict, dataclass
from weakref import WeakValueDictionary

from langgraph.types import interrupt
from sqlalchemy.exc import IntegrityError

from app.db.models import Conversation, ConversationStatus, Ticket, TicketStatus, TicketType


class TicketDraftConflict(RuntimeError):
    """The stable call identity already exists with different business content."""


@dataclass(frozen=True)
class TicketDraft:
    confirmation_id: str
    conversation_id: int
    tool_call_id: str
    ticket_type: str
    description: str
    request_message_id: int

    def __post_init__(self):
        if (self.conversation_id <= 0 or self.request_message_id <= 0
                or not self.tool_call_id.strip() or not self.description.strip()
                or not self.confirmation_id.strip()):
            raise ValueError("工单草稿缺少必填数据")
        TicketType(self.ticket_type)

    @classmethod
    def create(
        cls,
        *,
        conversation_id: int,
        tool_call_id: str,
        ticket_type: str,
        description: str,
        request_message_id: int,
    ) -> "TicketDraft":
        canonical = "\x1f".join(
            (
                str(conversation_id),
                tool_call_id,
                ticket_type,
                description,
                str(request_message_id),
            )
        )
        confirmation_id = "TC" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:30]
        return cls(
            confirmation_id=confirmation_id,
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            ticket_type=ticket_type,
            description=description,
            request_message_id=request_message_id,
        )


def ticket_no_for(draft: TicketDraft) -> str:
    identity = f"{draft.conversation_id}:{draft.tool_call_id}"
    return "T8" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:30]


def confirm_ticket(draft: TicketDraft, context) -> bool:
    """Pause for a trusted HTTP resume decision and reject mismatched responses."""
    if (draft.conversation_id != context.conversation_id
            or draft.tool_call_id != context.tool_call_id):
        raise PermissionError("工单草稿与执行上下文不一致")
    response = interrupt(
        {
            "type": "ticket_confirmation",
            "confirmation_id": draft.confirmation_id,
            "draft": asdict(draft),
        }
    )
    if not isinstance(response, dict):
        return False
    return (
        response.get("decision") == "confirm"
        and response.get("confirmation_id") == draft.confirmation_id
    )


def draft_from_call(arguments: dict, context) -> TicketDraft:
    evidence = context.ticket_request or {}
    if getattr(evidence, "to_dict", None):
        evidence = evidence.to_dict()
    if not isinstance(evidence, dict) or evidence.get("status") != "active":
        raise PermissionError("没有有效的用户建单请求证据")
    return TicketDraft.create(
        conversation_id=int(context.conversation_id),
        tool_call_id=str(context.tool_call_id),
        ticket_type=str(arguments.get("ticket_type") or ""),
        description=str(arguments.get("description") or "").strip(),
        request_message_id=int(evidence["message_id"]),
    )


async def confirm_ticket_call(definition, arguments: dict, context) -> bool:
    """ToolEngine confirmation adapter; only server-owned context supplies evidence."""
    return confirm_ticket(draft_from_call(arguments, context), context)


def _matches(row: Ticket, draft: TicketDraft) -> bool:
    ticket_type = row.ticket_type.value if hasattr(row.ticket_type, "value") else row.ticket_type
    return (
        row.conversation_id == draft.conversation_id
        and row.description == draft.description
        and ticket_type == draft.ticket_type
    )


_ticket_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
_ticket_locks_guard = asyncio.Lock()


async def _lock_for(ticket_no: str) -> asyncio.Lock:
    async with _ticket_locks_guard:
        return _ticket_locks.setdefault(ticket_no, asyncio.Lock())


async def create_ticket_once(session_factory, draft: TicketDraft, *, escalate: bool = True) -> str:
    """Insert one stable ticket; an identical replay returns the existing number."""
    ticket_no = ticket_no_for(draft)
    lock = await _lock_for(ticket_no)
    async with lock:
        async with session_factory() as session:
            existing = await session.get(Ticket, ticket_no)
            if existing is not None:
                if not _matches(existing, draft):
                    raise TicketDraftConflict("同一工具调用已存在内容不同的工单")
                return ticket_no

            row = Ticket(
                ticket_no=ticket_no,
                conversation_id=draft.conversation_id,
                description=draft.description,
                ticket_type=TicketType(draft.ticket_type),
                status=TicketStatus.PENDING,
            )
            try:
                session.add(row)
                if escalate:
                    conversation = await session.get(Conversation, draft.conversation_id)
                    if conversation is not None:
                        conversation.status = ConversationStatus.ESCALATED
                await session.commit()
                return ticket_no
            except IntegrityError as error:
                await session.rollback()
                integrity_error = error

        async with session_factory() as session:
            existing = await session.get(Ticket, ticket_no)
            if existing is None:
                raise integrity_error
            if not _matches(existing, draft):
                raise TicketDraftConflict("同一工具调用已存在内容不同的工单")
            return ticket_no
