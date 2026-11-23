from pydantic import BaseModel, Field

from multi_agent_chat.chatbot.estados.schemas import ResultadoProjeto
from multi_agent_chat.models.conversation import ConversationStatus
from multi_agent_chat.schemas.conversations import MAX_LIMIT, MIN_LIMIT
from multi_agent_chat.schemas.messages import MessageResponse


class ChatRequest(BaseModel):
    content: str = ''
    finish: bool = False


class ReopenRequest(BaseModel):
    additional_rounds: int = Field(default=1, ge=MIN_LIMIT, le=MAX_LIMIT)
    additional_interviews: int = Field(default=1, ge=0, le=MAX_LIMIT)
    content: str = ''


class ChatResponse(BaseModel):
    conversation_id: int
    status: ConversationStatus
    user_message: MessageResponse
    assistant_message: MessageResponse
    question: str | None = None
    question_index: int | None = None
    question_total: int | None = None
    specification: ResultadoProjeto | None = None


class SpecificationResponse(BaseModel):
    conversation_id: int
    status: ConversationStatus
    specification: ResultadoProjeto
    markdown: str
