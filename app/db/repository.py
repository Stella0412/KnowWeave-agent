from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Conversation,
    ConversationStatus,
    FAQ,
    KnowledgeChunk,
    Message,
    MessageRole,
    QAExtractionStaging,
    StagingStatus,
    Ticket,
    TicketStatus,
    TicketType,
    VectorizeStatus,
)


class ChatRepository:
    """会话/消息/工单/FAQ 的读写封装。方法只改 session，由调用方 commit。"""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create_conversation(self, user_id: str = "anonymous") -> int:
        conv = Conversation(user_id=user_id)
        self._session.add(conv)
        await self._session.flush()
        return conv.id

    async def get_conversation(self, conversation_id: int) -> Conversation | None:
        return await self._session.get(Conversation, conversation_id)

    async def set_conversation_status(
        self, conversation_id: int, status: ConversationStatus
    ) -> None:
        conv = await self.get_conversation(conversation_id)
        if conv is not None:
            conv.status = status

    async def add_message(
        self,
        conversation_id: int,
        *,
        role: MessageRole | str,
        content: str | None = None,
        tool_calls: list | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        self._session.add(
            Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
            )
        )

    async def list_messages(self, conversation_id: int) -> list[Message]:
        result = await self._session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list(result.scalars())

    async def list_conversations(self) -> list[Conversation]:
        result = await self._session.execute(select(Conversation).order_by(Conversation.id))
        return list(result.scalars())

    async def search_faq(self, keyword: str) -> list[FAQ]:
        """[已不在召回链路] ch02 的关键词查表；ch03 起 query_faq 走向量检索。保留仅为 ch02 遗留。"""
        like = f"%{keyword}%"
        result = await self._session.execute(
            select(FAQ).where(FAQ.question.like(like) | FAQ.answer.like(like))
        )
        return list(result.scalars())

    async def create_ticket(
        self, conversation_id: int, description: str, ticket_type: TicketType | str
    ) -> str:
        prefix = "T" + datetime.now().strftime("%Y%m%d")
        result = await self._session.execute(
            select(func.count())
            .select_from(Ticket)
            .where(Ticket.ticket_no.like(prefix + "%"))
        )
        seq = (result.scalar() or 0) + 1
        ticket_no = f"{prefix}{seq:03d}"
        self._session.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
                status=TicketStatus.PENDING,
            )
        )
        return ticket_no


class KnowledgeRepository:
    """知识库 chunk 与 QA 暂存的读写封装。方法只改 session，由调用方 commit。"""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def insert_chunks(
        self, chunks: list, *, status: VectorizeStatus = VectorizeStatus.PENDING
    ) -> list[int]:
        rows = [
            KnowledgeChunk(
                category=c.category,
                questions=c.questions,
                answer=c.answer,
                section_path=c.section_path,
                content_type=c.content_type,
                is_key_clause=c.is_key_clause,
                vectorize_status=status,
            )
            for c in chunks
        ]
        self._session.add_all(rows)
        await self._session.flush()
        return [r.id for r in rows]

    async def link_neighbors(self, ids: list[int]) -> None:
        """按给定顺序串成双向链表；文档/批次的边界块指针置 None。"""
        for i, cid in enumerate(ids):
            row = await self._session.get(KnowledgeChunk, cid)
            if row is None:
                continue
            row.prev_chunk_id = ids[i - 1] if i > 0 else None
            row.next_chunk_id = ids[i + 1] if i + 1 < len(ids) else None

    async def doc_root_exists(self, root: str) -> bool:
        stmt = select(KnowledgeChunk.id).where(
            (KnowledgeChunk.section_path == root)
            | KnowledgeChunk.section_path.like(root + " > %")
        )
        return (await self._session.execute(stmt.limit(1))).first() is not None

    async def delete_by_doc_root(self, root: str) -> list[int]:
        stmt = select(KnowledgeChunk.id).where(
            (KnowledgeChunk.section_path == root)
            | KnowledgeChunk.section_path.like(root + " > %")
        )
        ids = list((await self._session.execute(stmt)).scalars())
        if ids:
            for cid in ids:
                row = await self._session.get(KnowledgeChunk, cid)
                if row is not None:
                    await self._session.delete(row)
            await self._session.flush()
        return ids

    async def list_pending(self, limit: int = 200) -> list[KnowledgeChunk]:
        stmt = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == VectorizeStatus.PENDING)
            .order_by(KnowledgeChunk.id)
            .limit(limit)
        )
        return list((await self._session.execute(stmt)).scalars())

    async def mark_vectorized(self, chunk_id: int, vector_id: str) -> None:
        row = await self._session.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.vector_id = vector_id
            row.vectorize_status = VectorizeStatus.DONE

    async def get_by_ids(self, ids: list[int]) -> list[KnowledgeChunk]:
        if not ids:
            return []
        rows = (
            await self._session.execute(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids)))
        ).scalars()
        by_id = {r.id: r for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    async def list_existing_qa(self) -> list[tuple[str, str]]:
        rows = (
            await self._session.execute(
                select(KnowledgeChunk.questions, KnowledgeChunk.answer)
            )
        ).all()
        return [(q, a) for q, a in rows]

    async def count_by_status(self, status: VectorizeStatus) -> int:
        stmt = (
            select(func.count())
            .select_from(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == status)
        )
        return (await self._session.execute(stmt)).scalar() or 0

    async def clear_batch(self, batch_no: str) -> None:
        rows = (
            await self._session.execute(
                select(QAExtractionStaging).where(QAExtractionStaging.batch_no == batch_no)
            )
        ).scalars()
        for row in rows:
            await self._session.delete(row)
        await self._session.flush()

    async def add_staging(
        self, batch_no: str, source_ref: str | None, question: str, answer: str
    ) -> None:
        self._session.add(
            QAExtractionStaging(
                batch_no=batch_no, source_ref=source_ref, question=question, answer=answer
            )
        )

    async def list_staging(
        self, batch_no: str | None, status: StagingStatus
    ) -> list[QAExtractionStaging]:
        stmt = select(QAExtractionStaging).where(QAExtractionStaging.status == status)
        if batch_no is not None:
            stmt = stmt.where(QAExtractionStaging.batch_no == batch_no)
        return list(
            (await self._session.execute(stmt.order_by(QAExtractionStaging.id))).scalars()
        )

    async def set_staging_status(self, ids: list[int], status: StagingStatus) -> None:
        for sid in ids:
            row = await self._session.get(QAExtractionStaging, sid)
            if row is not None:
                row.status = status
