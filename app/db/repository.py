from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Conversation,
    ConversationStatus,
    FAQ,
    Message,
    MessageRole,
    Ticket,
    TicketStatus,
    TicketType,
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

    async def search_faq(self, keyword: str) -> list[FAQ]:
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
