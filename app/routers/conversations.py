from fastapi import APIRouter, HTTPException, Request

from app.db.engine import SessionFactory
from app.db.repository import ChatRepository

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("")
async def list_conversations(user_id: str = "anonymous") -> dict:
    async with SessionFactory() as session:
        repo = ChatRepository(session)
        rows = await repo.list_conversations_for_user(user_id)
        items = []
        for row in rows:
            preview = (await repo.first_user_content(row.id)).strip()[:40]
            count = await repo.count_ui_messages(row.id)
            items.append({
                "id": row.id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "preview": preview,
                "summarized": row.summary_upto_msg_id is not None,
                "message_count": count,
            })
    return {"conversations": items}


@router.get("/{conversation_id}/messages")
async def list_messages(conversation_id: int, request: Request, user_id: str = "anonymous") -> dict:
    async with SessionFactory() as session:
        repo = ChatRepository(session)
        conv = await repo.get_conversation(conversation_id)
        if conv is None or conv.user_id != user_id:
            raise HTTPException(status_code=404, detail="会话不存在")
        rows = await repo.list_ui_messages(conversation_id)
    service = getattr(request.app.state, "chat_service", None)
    pending_action = None
    if service is not None:
        pending_action = await service.pending_action_for(conversation_id)
    return {
        "conversation_id": conversation_id,
        "pending_action": pending_action,
        "messages": [
            {
                "id": row.id,
                "role": row.role.value,
                "content": row.content or "",
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ],
    }
