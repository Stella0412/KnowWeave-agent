from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Conversation,
    ConversationStatus,
    ConversationSummary,
    FaithCase,
    FaithCaseStatus,
    FAQ,
    KnowledgeChunk,
    LowConfidenceQuestion,
    LowConfidenceSource,
    Message,
    MessageRole,
    Order,
    QAExtractionStaging,
    RefundRequest,
    RefundStatus,
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

    async def get_conversation_for_update(self, conversation_id: int) -> Conversation | None:
        result = await self._session.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .with_for_update()
        )
        return result.scalars().first()

    async def list_summaries(self, conversation_id: int) -> list[ConversationSummary]:
        result = await self._session.execute(
            select(ConversationSummary)
            .where(ConversationSummary.conversation_id == conversation_id)
            .order_by(ConversationSummary.seq)
        )
        return list(result.scalars())

    async def next_summary_seq(self, conversation_id: int) -> int:
        result = await self._session.execute(
            select(func.max(ConversationSummary.seq)).where(
                ConversationSummary.conversation_id == conversation_id
            )
        )
        return int(result.scalar() or 0) + 1

    async def add_summary(
        self,
        conversation_id: int,
        *,
        seq: int,
        from_msg_id: int,
        upto_msg_id: int,
        content: str,
    ) -> ConversationSummary:
        row = ConversationSummary(
            conversation_id=conversation_id,
            seq=seq,
            from_msg_id=from_msg_id,
            upto_msg_id=upto_msg_id,
            content=content,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def touch_conversation(self, conversation_id: int) -> None:
        conv = await self.get_conversation(conversation_id)
        if conv is not None:
            conv.updated_at = datetime.now()

    async def set_layer1_from(self, conversation_id: int, msg_id: int) -> None:
        conv = await self.get_conversation(conversation_id)
        if conv is not None:
            conv.layer1_from_msg_id = msg_id

    async def list_conversations_for_user(self, user_id: str = "anonymous") -> list[Conversation]:
        result = await self._session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        )
        return list(result.scalars())

    async def first_user_content(self, conversation_id: int) -> str:
        result = await self._session.execute(
            select(Message.content)
            .where(Message.conversation_id == conversation_id, Message.role == MessageRole.USER)
            .order_by(Message.id)
            .limit(1)
        )
        return result.scalar() or ""

    async def list_ui_messages(self, conversation_id: int) -> list[Message]:
        result = await self._session.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role.in_([MessageRole.USER, MessageRole.ASSISTANT]),
                Message.content.is_not(None),
                func.trim(Message.content) != "",
            )
            .order_by(Message.id)
        )
        return list(result.scalars())

    async def count_ui_messages(self, conversation_id: int) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role.in_([MessageRole.USER, MessageRole.ASSISTANT]),
                Message.content.is_not(None),
                func.trim(Message.content) != "",
            )
        )
        return int(result.scalar() or 0)

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
    ) -> Message:
        row = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        self._session.add(row)
        await self._session.flush()
        return row

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
        # ch08 起建单改走 ticket_confirmation.create_ticket_once（稳定 T8+sha256 工单号）。
        # 本方法已无生产调用方，仅 tests/test_repository.py 的遗留单测仍依赖它验证仓库层
        # 自己的旧工单号生成逻辑（T+日期+序号），故保留；新代码不要再引用。

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
                row_category=getattr(c, "row_category", None),
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


class LowConfidenceRepository:
    """低置信度问题池读写。方法只改 session，由调用方 commit。"""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self,
        conversation_id: int | None,
        raw_question: str,
        source: LowConfidenceSource | str,
        reason: str | None,
    ) -> LowConfidenceQuestion:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id,
            raw_question=raw_question,
            source=source,
            reason=reason,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_questions(
        self, source: str | None = None, limit: int = 50
    ) -> list[LowConfidenceQuestion]:
        stmt = select(LowConfidenceQuestion)
        if source is not None:
            stmt = stmt.where(LowConfidenceQuestion.source == source)
        stmt = stmt.order_by(LowConfidenceQuestion.id.desc()).limit(limit)
        return list((await self._session.execute(stmt)).scalars())

    async def count_by_source(self) -> dict[str, int]:
        stmt = select(LowConfidenceQuestion.source, func.count()).group_by(
            LowConfidenceQuestion.source
        )
        return {
            (row.source.value if hasattr(row.source, "value") else str(row.source)): int(row[1])
            for row in (await self._session.execute(stmt)).all()
        }


class FaithCaseRepository:
    """编造个案台账读写。一题一行，复发不新增行只累加 seen_count。"""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def upsert_case(
        self,
        *,
        eval_id: str,
        bucket: str,
        query: str,
        strategy: str,
        answer: str,
        reason: str,
        citations: list | None,
        judge_model: str | None,
    ) -> FaithCase:
        now = datetime.now()
        row = (
            await self._session.execute(
                select(FaithCase).where(FaithCase.eval_id == eval_id)
            )
        ).scalar_one_or_none()
        if row is None:
            row = FaithCase(
                eval_id=eval_id, bucket=bucket, query=query, strategy=strategy,
                answer=answer, reason=reason, citations=citations,
                judge_model=judge_model, status=FaithCaseStatus.OPEN,
                seen_count=1, first_seen_at=now, last_seen_at=now,
            )
            self._session.add(row)
            await self._session.flush()
            return row

        # 同一道题再次被判编造：更新成最近一次的快照，累加次数
        row.bucket = bucket
        row.query = query
        row.strategy = strategy
        row.answer = answer
        row.reason = reason
        row.citations = citations
        row.judge_model = judge_model
        row.seen_count += 1
        row.last_seen_at = now
        # 已经处置过的又被判出来 → 那是复发，退回未解决；resolved_at 保留，用来标「复发」
        if row.status is not FaithCaseStatus.OPEN:
            row.status = FaithCaseStatus.OPEN
            row.resolution = None
        await self._session.flush()
        return row

    async def list_cases(
        self, status: FaithCaseStatus | str | None = None, limit: int = 100
    ) -> list[FaithCase]:
        stmt = select(FaithCase)
        if status is not None:
            stmt = stmt.where(FaithCase.status == status)
        stmt = stmt.order_by(FaithCase.last_seen_at.desc(), FaithCase.id.desc()).limit(limit)
        return list((await self._session.execute(stmt)).scalars())

    async def get_cases(self, eval_ids: list[str]) -> dict[str, FaithCase]:
        """按题号批量取台账行；当前轮只渲染这些行，跨轮累计另行统计。"""
        if not eval_ids:
            return {}
        stmt = select(FaithCase).where(FaithCase.eval_id.in_(eval_ids))
        rows = list((await self._session.execute(stmt)).scalars())
        return {row.eval_id: row for row in rows}

    async def count_by_status(self) -> dict[str, int]:
        stmt = select(FaithCase.status, func.count()).group_by(FaithCase.status)
        counts: dict[str, int] = {}
        for status, count in (await self._session.execute(stmt)).all():
            key = status.value if hasattr(status, "value") else str(status)
            counts[key] = int(count)
        return counts

    async def set_status(
        self,
        eval_id: str,
        status: FaithCaseStatus | str,
        resolution: str | None = None,
    ) -> FaithCase | None:
        row = (
            await self._session.execute(select(FaithCase).where(FaithCase.eval_id == eval_id))
        ).scalar_one_or_none()
        if row is None:
            return None
        row.status = status
        if str(status) in (FaithCaseStatus.RESOLVED.value, FaithCaseStatus.NOT_NEEDED.value) or (
            isinstance(status, FaithCaseStatus) and status is not FaithCaseStatus.OPEN
        ):
            row.resolution = resolution
            row.resolved_at = datetime.now()
        else:
            # 退回未解决：处置说明清空（空着的处置在台账上等于没有交代）
            row.resolution = None
        await self._session.flush()
        return row


class OrderRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def list_for_user(
        self, user_id: str = "anonymous", limit: int = 20
    ) -> list[Order]:
        result = await self._session.execute(
            select(Order)
            .where(Order.user_id == user_id)
            .order_by(Order.order_no)
            .limit(limit)
        )
        return list(result.scalars())

    async def get(self, order_no: str, *, for_update: bool = False) -> Order | None:
        """按订单号取单。`for_update=True` 在支持行锁的库上锁住该行，
        用于「查 pending → 建退款单」这类读后写流程，避免并发下重复建单。"""
        key = (order_no or "").strip()
        if not for_update:
            return await self._session.get(Order, key)
        result = await self._session.execute(
            select(Order).where(Order.order_no == key).with_for_update()
        )
        return result.scalars().first()

    async def get_by_tracking(self, tracking_no: str) -> Order | None:
        """按物流单号取单；物流工具用它的签收时间对齐 mock 轨迹。"""
        key = (tracking_no or "").strip()
        if not key:
            return None
        result = await self._session.execute(
            select(Order).where(Order.tracking_no == key)
        )
        return result.scalars().first()


class RefundRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_pending(
        self, conversation_id: int, order_no: str
    ) -> RefundRequest | None:
        result = await self._session.execute(
            select(RefundRequest).where(
                RefundRequest.conversation_id == conversation_id,
                RefundRequest.order_no == order_no,
                RefundRequest.status == RefundStatus.PENDING,
            )
        )
        return result.scalars().first()

    async def create(
        self,
        conversation_id: int,
        order_no: str,
        reason_category: str,
        note: str | None = None,
    ) -> RefundRequest:
        prefix = "R" + datetime.now().strftime("%Y%m%d")
        count = (
            await self._session.execute(
                select(func.count())
                .select_from(RefundRequest)
                .where(RefundRequest.refund_no.like(prefix + "%"))
            )
        ).scalar() or 0
        row = RefundRequest(
            refund_no=f"{prefix}{count + 1:03d}",
            conversation_id=conversation_id,
            order_no=order_no,
            reason_category=reason_category,
            note=note,
            status=RefundStatus.PENDING,
        )
        self._session.add(row)
        await self._session.flush()
        return row
