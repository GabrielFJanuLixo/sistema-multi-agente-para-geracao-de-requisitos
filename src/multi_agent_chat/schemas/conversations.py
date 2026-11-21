from datetime import datetime

from pydantic import BaseModel, Field

from multi_agent_chat.models import ConversationStatus
from multi_agent_chat.schemas.common import PartialUpdate

MIN_LIMIT = 1
MAX_LIMIT = 10
DEFAULT_LIMIT = 2


class ConversationCreate(BaseModel):
    title: str
    max_rounds: int = Field(default=DEFAULT_LIMIT, ge=MIN_LIMIT, le=MAX_LIMIT)
    max_interviews: int = Field(
        default=DEFAULT_LIMIT, ge=MIN_LIMIT, le=MAX_LIMIT
    )


class ConversationResponse(BaseModel):
    id: int
    user_id: int
    title: str
    created_at: datetime
    updated_at: datetime
    status: ConversationStatus
    max_rounds: int
    max_interviews: int


class ConversationUpdate(PartialUpdate):
    title: str | None = None
    max_rounds: int | None = Field(default=None, ge=MIN_LIMIT, le=MAX_LIMIT)
    max_interviews: int | None = Field(
        default=None, ge=MIN_LIMIT, le=MAX_LIMIT
    )


class ConversationReplace(ConversationCreate):
    pass
