from collections.abc import AsyncIterable
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from app.config import settings
from app.core.llm import build_chat_model, build_structured_model
from app.db.engine import SessionFactory
from app.db.repository import ChatRepository
from app.prompts.orchestrator import IntentOutput
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.tools.business import create_ticket_record

router = APIRouter(prefix="/chat", tags=["chat"])

# 懒实例化：避免 import 时用空 key 构造 ChatOpenAI 报错（ch01 Risk A）。
_chat_service: ChatService | None = None


class TicketRequest(BaseModel):
    session_id: str
    ticket_type: Literal["售后", "投诉", "咨询"]
    description: str = Field(..., min_length=1, max_length=1000)


def _get_chat_service(request: Request) -> ChatService:
    global _chat_service
    if _chat_service is None:
        graph = getattr(request.app.state, "graph", None)
        if graph is None:
            raise RuntimeError("图未初始化：请通过 app.main lifespan 启动")
        _chat_service = ChatService(
            build_chat_model(settings),
            build_structured_model(settings, IntentOutput, temperature=0.0),
            SessionFactory,
            graph,
            token_budget=settings.token_budget,
            tool_timeout=settings.tool_timeout,
            tool_retries=settings.tool_retries,
        )
    return _chat_service


@router.post("", response_class=EventSourceResponse)
async def chat(req: ChatRequest, request: Request) -> AsyncIterable[ServerSentEvent]:
    async for event in _get_chat_service(request).stream(req.session_id, req.message):
        yield ServerSentEvent(data=event)


@router.post("/agent")
async def chat_agent(req: ChatRequest, request: Request) -> dict:
    try:
        return await _get_chat_service(request).run(req.session_id, req.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ticket")
async def create_ticket_endpoint(req: TicketRequest) -> dict:
    try:
        conversation_id = int(req.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="session_id 必须是数字") from exc

    async with SessionFactory() as session:
        conv = await ChatRepository(session).get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    ticket_no = await create_ticket_record(
        SessionFactory,
        conversation_id,
        req.description,
        req.ticket_type,
        escalate=False,
    )
    return {"ticket_no": ticket_no}
