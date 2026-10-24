from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1)


class ResumeRequest(BaseModel):
    session_id: str
    order_no: str = Field(..., min_length=1)


class TicketResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    confirmation_id: str
    decision: Literal["confirm", "cancel"]


class RefundRequest(BaseModel):
    session_id: str
    order_no: str = Field(..., min_length=1)
    reason_category: str
    note: str | None = Field(default=None, max_length=500)
