from app.db.base import Base
from app.db.models import (
    Conversation,
    ConversationStatus,
    FAQ,
    Message,
    Ticket,
    TicketType,
)


def test_tables_registered():
    assert set(Base.metadata.tables) == {"conversations", "messages", "faq", "tickets"}


def test_conversation_columns():
    cols = {c.name for c in Conversation.__table__.columns}
    assert {"id", "user_id", "status", "created_at", "updated_at"} <= cols


def test_message_columns():
    cols = {c.name for c in Message.__table__.columns}
    assert {"id", "conversation_id", "role", "content", "tool_calls", "tool_call_id", "created_at"} <= cols


def test_ticket_columns():
    cols = {c.name for c in Ticket.__table__.columns}
    assert {"ticket_no", "conversation_id", "description", "ticket_type", "status", "created_at"} <= cols


def test_faq_columns():
    cols = {c.name for c in FAQ.__table__.columns}
    assert {"id", "question", "answer", "category", "created_at", "updated_at"} <= cols


def test_enum_values():
    assert {m.value for m in ConversationStatus} == {"进行中", "已转人工", "已结束"}
    assert {m.value for m in TicketType} == {"售后", "投诉", "咨询"}
