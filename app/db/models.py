import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
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
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_upto_msg_id: Mapped[int | None] = mapped_column(BIGINT, nullable=True)
    layer1_from_msg_id: Mapped[int | None] = mapped_column(BIGINT, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="uk_conversation_seq"),
        Index("idx_conversation_upto", "conversation_id", "upto_msg_id"),
    )

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("conversations.id"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    from_msg_id: Mapped[int] = mapped_column(BIGINT)
    upto_msg_id: Mapped[int] = mapped_column(BIGINT)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


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


class RefundStatus(str, enum.Enum):
    PENDING = "待审核"
    APPROVED = "已通过"
    REJECTED = "已拒绝"


class Order(Base):
    __tablename__ = "orders"

    order_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="anonymous")
    product_name: Mapped[str] = mapped_column(String(255))
    product_type: Mapped[str] = mapped_column(String(32), default="普通")
    amount: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16))
    tracking_no: Mapped[str | None] = mapped_column(String(64), nullable=True)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    signed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class RefundRequest(Base):
    __tablename__ = "refund_requests"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    refund_no: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("conversations.id"), index=True
    )
    order_no: Mapped[str] = mapped_column(
        String(32), ForeignKey("orders.order_no"), index=True
    )
    reason_category: Mapped[str] = mapped_column(String(32))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[RefundStatus] = mapped_column(
        Enum(RefundStatus, values_callable=lambda e: [m.value for m in e]),
        default=RefundStatus.PENDING,
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
    # 分类维度表按行拆时该行分类列的取值（ch04 新增；⚠️ 唯一一列偏离 ch03「不加列不改列」，
    # 已获用户批准。必须落库：值在切分时才知道，chunk 切好后无法从正文反推，只写 Milvus 重建即丢）
    row_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
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


class LowConfidenceSource(str, enum.Enum):
    """入池入口。user_feedback 本章不写，留给 ch09 的满意度反馈落库。"""

    RETRIEVAL_LOW_CONF = "retrieval_low_conf"
    SELF_CHECK = "self_check"
    USER_FEEDBACK = "user_feedback"


class AnswerRating(str, enum.Enum):
    UP = "up"
    DOWN = "down"


class LowConfidenceStatus(str, enum.Enum):
    PENDING = "待处理"
    MERGED = "已归并"
    CLOSED = "已关闭"
    IGNORED = "已忽略"


class TurnStatus(str, enum.Enum):
    OK = "ok"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class FaithCaseStatus(str, enum.Enum):
    OPEN = "未解决"
    RESOLVED = "已解决"
    NOT_NEEDED = "无需解决"


class LowConfidenceQuestion(Base):
    """低置信度问题池：答不出来的原话落这里，是 ch09 数据飞轮的入口。"""

    __tablename__ = "low_confidence_questions"
    __table_args__ = (
        UniqueConstraint("message_id", "source", name="uk_message_source"),
        Index("idx_fingerprint_root", "fingerprint", "merged_into_id"),
        Index("idx_status_last_seen", "status", "last_seen_at"),
        Index("idx_merged_into", "merged_into_id"),
    )

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        BIGINT, ForeignKey("conversations.id"), nullable=True, index=True
    )
    message_id: Mapped[int | None] = mapped_column(BIGINT, nullable=True)
    raw_question: Mapped[str] = mapped_column(Text, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, server_default="")
    merged_into_id: Mapped[int | None] = mapped_column(BIGINT, nullable=True)
    duplicate_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    status: Mapped[LowConfidenceStatus] = mapped_column(
        Enum(LowConfidenceStatus, values_callable=lambda e: [m.value for m in e]),
        nullable=False, default=LowConfidenceStatus.PENDING,
        server_default=LowConfidenceStatus.PENDING.value,
    )
    source: Mapped[LowConfidenceSource] = mapped_column(
        Enum(LowConfidenceSource, values_callable=lambda e: [m.value for m in e]), index=True
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    knowledge_chunk_id: Mapped[int | None] = mapped_column(BIGINT, nullable=True)
    resolution: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class AnswerFeedback(Base):
    """👍/👎 反馈落库：一条助手消息只有一个最终评价，与飞轮/审计表解耦。"""

    __tablename__ = "answer_feedback"
    __table_args__ = (
        UniqueConstraint("message_id", name="uk_feedback_message"),
        Index("idx_feedback_conversation", "conversation_id"),
        Index("idx_feedback_rating_created", "rating", "created_at"),
    )

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(BIGINT, nullable=False)
    message_id: Mapped[int] = mapped_column(BIGINT, nullable=False, unique=True)
    rating: Mapped[AnswerRating] = mapped_column(
        Enum(AnswerRating, values_callable=lambda e: [m.value for m in e]), nullable=False
    )
    comment: Mapped[str | None] = mapped_column(String(300), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class TurnMetrics(Base):
    """每轮链路指标：route/intent/状态/token/成本/耗时，不含 prompt/answer 正文。"""

    __tablename__ = "turn_metrics"
    __table_args__ = (
        UniqueConstraint("turn_token", name="uk_turn_token"),
        Index("idx_turn_created", "created_at"),
        Index("idx_turn_conversation", "conversation_id"),
    )

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(BIGINT, nullable=False)
    turn_token: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    route: Mapped[str | None] = mapped_column(String(32), nullable=True)
    intent: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[TurnStatus] = mapped_column(
        Enum(TurnStatus, values_callable=lambda e: [m.value for m in e]), nullable=False
    )
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    llm_calls: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    usage_estimated: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("0"))
    model_usage: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    stages: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cost_cny: Mapped[float] = mapped_column(Numeric(12, 6), nullable=False, server_default=text("0"))
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    first_token_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    refusal_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    citations_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class FaithCase(Base):
    """忠实度编造个案台账：一题一行，跨轮追溯同一道题反复被判编造。"""

    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(16), unique=True)
    bucket: Mapped[str] = mapped_column(String(24))
    query: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str] = mapped_column(String(24), default="hybrid_rerank")
    answer: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    # 这一轮喂给模型的 Top-K 证据全集快照（不只存被引用的那几条：手里有哪几条、实际引了哪几条
    # 都是判断依据），形状与 SSE citations 载荷一致
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[FaithCaseStatus] = mapped_column(
        Enum(FaithCaseStatus, values_callable=lambda e: [m.value for m in e]),
        default=FaithCaseStatus.OPEN,
        index=True,
    )
    seen_count: Mapped[int] = mapped_column(Integer, default=1)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AuditStatus(str, enum.Enum):
    """工具调用终态，五个中文取值与用户 DDL 的 ENUM 一致。"""

    SUCCESS = "成功"
    FAILURE = "失败"
    TIMEOUT = "超时"
    VALIDATION_BLOCKED = "校验拦下"
    PERMISSION_DENIED = "权限拒绝"


class ToolSource(str, enum.Enum):
    BUILTIN = "builtin"
    MCP = "mcp"


class ToolAuditLog(Base):
    """工具调用审计留痕：内置与 MCP 工具每次逻辑调用落一条。

    审计事务独立于业务事务，因此 conversation_id 不挂外键、只建普通索引，
    避免被引用约束拦住审计写入。
    """

    __tablename__ = "tool_audit_logs"
    __table_args__ = (
        Index("idx_conversation_id", "conversation_id"),
        Index("idx_tool_name", "tool_name"),
        Index("idx_status", "status"),
    )

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BIGINT, nullable=True)
    turn_token: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_source: Mapped[ToolSource] = mapped_column(
        Enum(ToolSource, values_callable=lambda e: [m.value for m in e]), nullable=False
    )
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[Any] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[AuditStatus] = mapped_column(
        Enum(AuditStatus, values_callable=lambda e: [m.value for m in e]), nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class TopicClassification(Base):
    """ch10 旁路主题归类结果：一个根问题一行，labels 为 17 类中的命中项。"""

    __tablename__ = "topic_classifications"
    __table_args__ = (
        UniqueConstraint("question_id", name="uk_question_id"),
    )

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("low_confidence_questions.id"), nullable=False
    )
    labels: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    classified_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
