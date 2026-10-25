import pytest
from sqlalchemy import select, text

from app.db.base import Base
from app.db.models import (
    Conversation,
    ConversationStatus,
    FAQ,
    KnowledgeChunk,
    Message,
    QAExtractionStaging,
    Ticket,
    TicketStatus,
    TicketType,
)


def test_tables_registered():
    assert set(Base.metadata.tables) == {
        "conversations",
        "messages",
        "faq",
        "tickets",
        "knowledge_chunks",
        "qa_extraction_staging",
        "low_confidence_questions",
        "faith_cases",
        "orders",
        "refund_requests",
        "conversation_summaries",
        "tool_audit_logs",
        "answer_feedback",
        "turn_metrics",
    }


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


@pytest.mark.asyncio
async def test_enum_columns_persist_chinese_literals(session_factory):
    async with session_factory() as session:
        conv = Conversation(user_id="u1", status=ConversationStatus.ACTIVE)
        session.add(conv)
        await session.commit()
        await session.refresh(conv)

        session.add(
            Ticket(
                ticket_no="T20261010001",
                conversation_id=conv.id,
                description="x",
                ticket_type=TicketType.AFTER_SALE,
            )
        )
        await session.commit()

        # ORM 往返：读回后仍是枚举成员，且 .value 是中文
        got = await session.get(Conversation, conv.id)
        assert got.status == ConversationStatus.ACTIVE
        assert got.status.value == "进行中"

        got_ticket = await session.get(Ticket, "T20261010001")
        assert got_ticket.ticket_type == TicketType.AFTER_SALE
        assert got_ticket.ticket_type.value == "售后"
        assert got_ticket.status == TicketStatus.PENDING

        # 落库的就是中文字面量，不是成员名 ACTIVE/AFTER_SALE
        conv_status = await session.scalar(
            text("select status from conversations where id = :i"), {"i": conv.id}
        )
        ticket_type = await session.scalar(
            text("select ticket_type from tickets where ticket_no = :t"),
            {"t": "T20261010001"},
        )
        assert conv_status == "进行中"
        assert ticket_type == "售后"


@pytest.mark.asyncio
async def test_knowledge_chunk_roundtrip(session_factory):
    from app.db.models import KnowledgeChunk, VectorizeStatus

    async with session_factory() as s:
        s.add(
            KnowledgeChunk(
                category="退货与运费政策",
                questions="邮费是多少\n运费怎么算",
                answer="单笔满 99 元包邮。",
                section_path="退货与运费政策 > 运费与邮费说明",
                content_type="policy",
                is_key_clause=True,
            )
        )
        await s.commit()
    async with session_factory() as s:
        row = (await s.execute(select(KnowledgeChunk))).scalars().one()
        assert row.vectorize_status is VectorizeStatus.PENDING
        assert row.is_key_clause is True
        assert row.vector_id is None
        assert row.prev_chunk_id is None and row.next_chunk_id is None
        assert "\n" in row.questions  # 多问法换行保留


@pytest.mark.asyncio
async def test_knowledge_chunk_status_literal(session_factory):
    """枚举列必须落 DDL 的字面量，而非成员名（ch02 Ruling 2 同款约束）。"""
    from app.db.models import KnowledgeChunk

    async with session_factory() as s:
        s.add(KnowledgeChunk(category="c", questions="q", answer="a"))
        await s.commit()
    async with session_factory() as s:
        stored = (await s.execute(text("SELECT vectorize_status FROM knowledge_chunks"))).scalar()
    assert stored == "pending"


@pytest.mark.asyncio
async def test_staging_roundtrip(session_factory):
    from app.db.models import QAExtractionStaging, StagingStatus

    async with session_factory() as s:
        s.add(
            QAExtractionStaging(
                batch_no="mine-20261012-001",
                source_ref="conversation:7",
                question="能开发票吗",
                answer="可以，下单时备注即可。",
            )
        )
        await s.commit()
    async with session_factory() as s:
        row = (await s.execute(select(QAExtractionStaging))).scalars().one()
        assert row.status is StagingStatus.EXTRACTED


def test_knowledge_chunk_has_row_category():
    cols = {c.name for c in KnowledgeChunk.__table__.columns}
    assert "row_category" in cols
    assert KnowledgeChunk.__table__.columns["row_category"].nullable is True
    assert KnowledgeChunk.__table__.columns["row_category"].type.length == 64


def test_low_confidence_question_columns():
    from app.db.models import LowConfidenceQuestion, LowConfidenceSource

    cols = {c.name for c in LowConfidenceQuestion.__table__.columns}
    assert {"id", "conversation_id", "raw_question", "source", "reason", "created_at"} <= cols
    assert {m.value for m in LowConfidenceSource} == {
        "retrieval_low_conf", "self_check", "user_feedback",
    }


def test_faith_case_columns_and_status_values():
    from app.db.models import FaithCase, FaithCaseStatus

    cols = {c.name for c in FaithCase.__table__.columns}
    assert {
        "id", "eval_id", "bucket", "query", "strategy", "answer", "reason", "citations",
        "judge_model", "status", "seen_count", "first_seen_at", "last_seen_at",
        "resolution", "resolved_at",
    } <= cols
    assert {m.value for m in FaithCaseStatus} == {"未解决", "已解决", "无需解决"}
    assert FaithCase.__table__.columns["eval_id"].unique is True


@pytest.mark.asyncio
async def test_faith_case_status_persists_chinese_literal(session_factory):
    """status 必须落 DDL 的中文字面量「未解决」而非枚举名 OPEN（values_callable 兜底）。"""
    from app.db.models import FaithCase, FaithCaseStatus

    async with session_factory() as session:
        session.add(FaithCase(
            eval_id="B03", bucket="B_model", query="型号?", answer="答案",
            reason="编在哪一句", status=FaithCaseStatus.OPEN,
        ))
        await session.commit()
    async with session_factory() as session:
        row = (await session.execute(select(FaithCase))).scalar_one()
        stored = await session.scalar(
            text("select status from faith_cases where eval_id = :e"), {"e": "B03"}
        )
    assert row.status is FaithCaseStatus.OPEN
    assert row.seen_count == 1
    assert stored == "未解决"
