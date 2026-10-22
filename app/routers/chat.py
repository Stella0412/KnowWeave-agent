import json
from collections.abc import AsyncIterable
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent, format_sse_event
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.core.llm import build_chat_model, build_structured_model
from app.db.engine import SessionFactory
from app.db.repository import ChatRepository, OrderRepository, RefundRepository
from app.prompts.orchestrator import CorefOutput, ExpansionOutput, IntentOutput
from app.schemas.chat import ChatRequest, RefundRequest, ResumeRequest
from app.services.chat_service import ChatService
from app.tools.business import REFUND_REASON_CATEGORIES, create_ticket_record

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
            build_structured_model(
                settings, IntentOutput, temperature=0.0,
                model_name=settings.intent_model or None,
            ),
            SessionFactory,
            graph,
            token_budget=settings.token_budget,
            tool_timeout=settings.tool_timeout,
            tool_retries=settings.tool_retries,
            coref_model=build_structured_model(settings, CorefOutput, temperature=0.0),
            expansion_model=build_structured_model(settings, ExpansionOutput, temperature=0.0),
            intent_mode=settings.intent_mode,
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


async def _sse_data(agen):
    """把 ChatService 的事件 dict 编码成与 /chat 一致的 SSE 线格式（data: JSON）。"""
    async for event in agen:
        yield format_sse_event(data_str=json.dumps(event, ensure_ascii=False))


@router.post("/resume")
async def chat_resume(req: ResumeRequest, request: Request) -> EventSourceResponse:
    service = _get_chat_service(request)
    try:
        await service.validate_resume(req.session_id, req.order_no)
    except ValueError as exc:
        message = str(exc)
        status = 400 if ("session_id" in message or "待选择" in message) else 404
        raise HTTPException(status_code=status, detail=message) from exc
    return EventSourceResponse(_sse_data(service.resume_stream(req.session_id, req.order_no)))


@router.post("/agent/resume")
async def chat_agent_resume(req: ResumeRequest, request: Request) -> dict:
    try:
        return await _get_chat_service(request).run_resume(req.session_id, req.order_no)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/refund")
async def refund(req: RefundRequest) -> dict:
    if req.reason_category not in REFUND_REASON_CATEGORIES:
        raise HTTPException(status_code=400, detail="退款原因不在固定类目内")
    try:
        conversation_id = int(req.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="session_id 必须是数字") from exc
    # 并发重复提交：先锁订单行，再做「查 pending → 建单」；refund_no 撞号时重试。
    for attempt in range(3):
        async with SessionFactory() as session:
            conv = await ChatRepository(session).get_conversation(conversation_id)
            order = await OrderRepository(session).get(req.order_no, for_update=True)
            if conv is None or order is None or order.user_id != conv.user_id:
                raise HTTPException(status_code=404, detail="会话或订单不存在")
            repo = RefundRepository(session)
            existing = await repo.get_pending(conversation_id, req.order_no)
            if existing is not None:
                return {"refund_no": existing.refund_no}
            try:
                row = await repo.create(
                    conversation_id, req.order_no, req.reason_category, req.note
                )
                await session.commit()
            except IntegrityError:
                await session.rollback()
                if attempt == 2:
                    raise HTTPException(
                        status_code=409, detail="退款单创建冲突，请稍后重试"
                    ) from None
                continue
            return {"refund_no": row.refund_no}


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
