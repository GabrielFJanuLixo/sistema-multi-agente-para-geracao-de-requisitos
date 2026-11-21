from datetime import datetime

from pydantic import BaseModel

from multi_agent_chat.models import AgentRole
from multi_agent_chat.schemas.common import PartialUpdate


class AgentCreate(BaseModel):
    name: str
    description: str
    model: str
    system_prompt: str
    temperature: float
    role: AgentRole | None = None


class AgentResponse(BaseModel):
    id: int
    name: str
    description: str
    model: str
    system_prompt: str
    temperature: float
    created_at: datetime
    updated_at: datetime
    role: AgentRole | None = None


class AgentUpdate(PartialUpdate):
    name: str | None = None
    description: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    role: AgentRole | None = None


class AgentReplace(AgentCreate):
    pass
