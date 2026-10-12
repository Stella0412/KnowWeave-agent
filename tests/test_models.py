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
