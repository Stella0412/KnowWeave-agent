from collections.abc import AsyncIterable

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.config import settings
from app.core.llm import build_chat_model
from app.db.engine import SessionFactory
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

# 懒实例化：避免 import 时用空 key 构造 ChatOpenAI 报错（ch01 Risk A）。
_chat_service: ChatService | None = None


def _get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService(
            build_chat_model(settings),
            SessionFactory,
            token_budget=settings.token_budget,
            tool_timeout=settings.tool_timeout,
            tool_retries=settings.tool_retries,
        )
    return _chat_service


@router.post("", response_class=EventSourceResponse)
async def chat(req: ChatRequest) -> AsyncIterable[ServerSentEvent]:
    async for event in _get_chat_service().stream(req.session_id, req.message):
        yield ServerSentEvent(data=event)
