import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Text, func
from sqlalchemy.orm import (
    Mapped,
    mapped_as_dataclass,
    mapped_column,
    relationship,
)

from multi_agent_chat.models.base import table_registry
from multi_agent_chat.models.conversation_agent import conversation_agents

if TYPE_CHECKING:
    from multi_agent_chat.models.agent import Agent
    from multi_agent_chat.models.message import Message
    from multi_agent_chat.models.user import User


class ConversationStatus(str, enum.Enum):
    IDLE = 'idle'
    WAITING_USER = 'waiting_user'
    COMPLETED = 'completed'
    COMPLETED_WITH_PENDING = 'completed_with_pending'


@mapped_as_dataclass(table_registry)
class Conversation:
    __tablename__ = 'conversations'

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE')
    )
    title: Mapped[str]
    status: Mapped[ConversationStatus] = mapped_column(
        default=ConversationStatus.IDLE
    )
    specification_json: Mapped[str | None] = mapped_column(Text, default=None)
    max_rounds: Mapped[int] = mapped_column(default=2)
    max_interviews: Mapped[int] = mapped_column(default=2)
    created_at: Mapped[datetime] = mapped_column(
        init=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        init=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    agents: Mapped[list['Agent']] = relationship(
        secondary=conversation_agents,
        back_populates='conversations',
        lazy='raise',
        init=False,
        repr=False,
        compare=False,
    )

    messages: Mapped[list['Message']] = relationship(
        back_populates='conversation',
        cascade='all, delete-orphan',
        lazy='raise',
        init=False,
        repr=False,
        compare=False,
    )

    user: Mapped['User'] = relationship(
        back_populates='conversations',
        lazy='raise',
        init=False,
        repr=False,
        compare=False,
    )
