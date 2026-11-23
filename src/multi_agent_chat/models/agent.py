import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Text, func
from sqlalchemy.orm import (
    Mapped,
    mapped_as_dataclass,
    mapped_column,
    relationship,
)

from multi_agent_chat.models.base import table_registry
from multi_agent_chat.models.conversation_agent import conversation_agents

if TYPE_CHECKING:
    from multi_agent_chat.models.conversation import Conversation
    from multi_agent_chat.models.message import Message


# Enum para representar o papel de um agente
class AgentRole(str, enum.Enum):
    INTERVIEWER = 'interviewer'
    BACKEND = 'backend'
    FRONTEND = 'frontend'
    QA = 'qa'


@mapped_as_dataclass(table_registry)
class Agent:
    __tablename__ = 'agents'

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    name: Mapped[str]
    description: Mapped[str]
    model: Mapped[str]
    system_prompt: Mapped[str] = mapped_column(Text)
    temperature: Mapped[float]
    role: Mapped[AgentRole | None] = mapped_column(default=None, unique=True)
    created_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        init=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    conversations: Mapped[list['Conversation']] = relationship(
        secondary=conversation_agents,
        back_populates='agents',
        lazy='raise',
        init=False,
        repr=False,
        compare=False,
    )

    messages: Mapped[list['Message']] = relationship(
        back_populates='agent',
        lazy='raise',
        init=False,
        repr=False,
        compare=False,
    )
