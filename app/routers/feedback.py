from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.feedback_service import FeedbackRejected

router = APIRouter(prefix="/api", tags=["feedback"])


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: int = Field(gt=0)
    message_id: int = Field(gt=0)
    rating: str
    comment: str | None = None

    @field_validator("rating")
    @classmethod
    def valid_rating(cls, value: str) -> str:
        if value not in ("up", "down"):
            raise ValueError("rating 只能是 up/down")
        return value

    @field_validator("comment")
    @classmethod
    def normalize_comment(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if len(value) > 300:
            raise ValueError("comment 不能超过 300 字符")
        return value


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest, request: Request) -> dict:
    service = getattr(request.app.state, "feedback_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="反馈服务未初始化")
    try:
        return await service.submit(
            conversation_id=req.conversation_id, message_id=req.message_id,
            rating=req.rating, comment=req.comment,
        )
    except FeedbackRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
