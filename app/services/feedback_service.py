"""ch09 反馈：先锁会话轮次，再校验最新消息、取快照、同事务写反馈与池。"""

import logging

from sqlalchemy.exc import IntegrityError

from app.db.models import AnswerFeedback, MessageRole
from app.db.repository import ChatRepository, FeedbackRepository
from app.services.chat_service import _lock_for_conversation

logger = logging.getLogger(__name__)


class FeedbackRejected(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FeedbackService:
    def __init__(self, session_factory, chat_service, flywheel):
        self._session_factory = session_factory
        self._chat = chat_service
        self._flywheel = flywheel

    async def submit(self, *, conversation_id, message_id, rating, comment):
        lock = await _lock_for_conversation(conversation_id)
        async with lock:
            return await self._submit_locked(
                conversation_id=conversation_id, message_id=message_id,
                rating=rating, comment=comment,
            )

    async def _submit_locked(self, *, conversation_id, message_id, rating, comment):
        async with self._session_factory() as session:
            chat_repo = ChatRepository(session)
            message = await chat_repo.get_message(message_id)
            if message is None or message.conversation_id != conversation_id:
                raise FeedbackRejected(404, "会话或消息不存在")
            if message.role is not MessageRole.ASSISTANT or not (message.content or "").strip():
                raise FeedbackRejected(422, "只能评价有内容的助手消息")
            user_message = await chat_repo.latest_user_before(conversation_id, message_id)
            if user_message is None or not (user_message.content or "").strip():
                raise FeedbackRejected(422, "找不到该回答对应的用户问题")

            latest_id, snapshot = await self._chat.latest_turn_snapshot(conversation_id)
            if latest_id != message_id:
                raise FeedbackRejected(409, "只能评价当前最新一条已完成助手消息")

            feedback_repo = FeedbackRepository(session)
            existing = await feedback_repo.get_by_message(message_id)
            if existing is not None:
                if existing.rating.value != str(rating):
                    raise FeedbackRejected(409, "该消息已有相反评价，不可修改")
                return await self._idempotent(session, existing, message_id)

            reason = "用户点踩"
            if comment:
                reason = (reason + "：" + comment)[:300]
            try:
                feedback = await feedback_repo.add(
                    conversation_id, message_id, rating, comment
                )
                result = None
                if str(rating) == "down":
                    result = await self._flywheel.record_event(
                        session, conversation_id=conversation_id,
                        raw_question=user_message.content, source="user_feedback",
                        reason=reason, message_id=message_id, citations=snapshot,
                    )
                await session.commit()
            except IntegrityError:
                await session.rollback()
                async with self._session_factory() as other:
                    existing = await FeedbackRepository(other).get_by_message(message_id)
                    if existing is None:
                        raise
                    if existing.rating.value != str(rating):
                        raise FeedbackRejected(409, "该消息已有相反评价，不可修改")
                    return await self._idempotent(other, existing, message_id)
        return {
            "feedback_id": feedback.id,
            "rating": str(rating),
            "pooled": result is not None,
            "cluster_id": result.root_id if result else None,
            "occurrences": result.duplicate_count if result else None,
            "citations_saved": bool(snapshot),
        }

    async def _idempotent(self, session, existing, message_id):
        event = await self._flywheel.find_message_event(
            session, message_id, "user_feedback"
        )
        root = await self._flywheel.resolve_root(session, event) if event else None
        return {
            "feedback_id": existing.id,
            "rating": existing.rating.value,
            "duplicate": True,
            "pooled": event is not None,
            "cluster_id": root.id if root else None,
            "occurrences": root.duplicate_count if root else None,
            "citations_saved": bool(event.citations) if event else False,
        }
