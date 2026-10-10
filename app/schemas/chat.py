from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1)
