import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# SQLite 只对 INTEGER PRIMARY KEY 自增；BigInteger 主键在测试库需降级为 Integer
BIGINT = BigInteger().with_variant(Integer, "sqlite")


class ConversationStatus(str, enum.Enum):
    ACTIVE = "进行中"
    ESCALATED = "已转人工"
    CLOSED = "已结束"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TicketType(str, enum.Enum):
    AFTER_SALE = "售后"
    COMPLAINT = "投诉"
    CONSULT = "咨询"


class TicketStatus(str, enum.Enum):
    PENDING = "待处理"
    DONE = "已处理"


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[ConversationStatus] = mapped_column(
        Enum(ConversationStatus, values_callable=lambda e: [m.value for m in e]),
        default=ConversationStatus.ACTIVE,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("conversations.id"), index=True
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, values_callable=lambda e: [m.value for m in e])
    )
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FAQ(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("conversations.id"), index=True
    )
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[TicketType] = mapped_column(
        Enum(TicketType, values_callable=lambda e: [m.value for m in e])
    )
    status: Mapped[TicketStatus] = mapped_column(
        Enum(TicketStatus, values_callable=lambda e: [m.value for m in e]),
        default=TicketStatus.PENDING,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class VectorizeStatus(str, enum.Enum):
    PENDING = "pending"
    DONE = "done"


class StagingStatus(str, enum.Enum):
    EXTRACTED = "extracted"
    KEPT = "kept"
    DISCARDED = "discarded"


class KnowledgeChunk(Base):
    """知识库 chunk 原文权威源。向量化文本 = category + questions + answer。"""

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    # 以下四类元数据只存不进向量
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT, ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT, ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True
    )
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[VectorizeStatus] = mapped_column(
        Enum(VectorizeStatus, values_callable=lambda e: [m.value for m in e]),
        default=VectorizeStatus.PENDING,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class QAExtractionStaging(Base):
    """历史对话抽 QA 的离线中转暂存表：分批抽取、整体去重，保留项入 knowledge_chunks。"""

    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64), index=True)
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[StagingStatus] = mapped_column(
        Enum(StagingStatus, values_callable=lambda e: [m.value for m in e]),
        default=StagingStatus.EXTRACTED,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
