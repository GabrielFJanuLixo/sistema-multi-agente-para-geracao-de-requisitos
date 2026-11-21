from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from fastapi.sse import EventSourceResponse

from multi_agent_chat.core.security import get_current_user
from multi_agent_chat.models.user import User
from multi_agent_chat.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SpecificationResponse,
)
from multi_agent_chat.services.chat import ChatService

router = APIRouter(prefix='/conversations/{conversation_id}', tags=['Chat'])
Service = Annotated[ChatService, Depends()]
CurrentUser = Annotated[User, Depends(get_current_user)]


# authenticated owner
@router.post(
    '/chat',
    status_code=HTTPStatus.OK,
    response_model=ChatResponse,
)
async def send_chat_message(
    conversation_id: int,
    chat_in: ChatRequest,
    service: Service,
    current_user: CurrentUser,
):
    return await service.send_message(
        conversation_id=conversation_id,
        user_id=current_user.id,
        content=chat_in.content,
        finish=chat_in.finish,
    )

# authenticated owner
@router.post('/chat/stream', status_code=HTTPStatus.OK)
async def stream_chat_message(
    conversation_id: int,
    chat_in: ChatRequest,
    service: Service,
    current_user: CurrentUser,
):
    events = await service.stream_message(
        conversation_id=conversation_id,
        user_id=current_user.id,
        content=chat_in.content,
        finish=chat_in.finish,
    )
    return EventSourceResponse(events)


# authenticated owner
@router.get(
    '/specification',
    status_code=HTTPStatus.OK,
    response_model=SpecificationResponse,
)
async def get_conversation_specification(
    conversation_id: int,
    service: Service,
    current_user: CurrentUser,
):
    return await service.get_specification(
        conversation_id=conversation_id,
        user_id=current_user.id,
    )


# authenticated owner
@router.get(
    '/specification/markdown',
    status_code=HTTPStatus.OK,
    response_class=PlainTextResponse,
)
async def get_conversation_specification_markdown(
    conversation_id: int,
    service: Service,
    current_user: CurrentUser,
):
    data = await service.get_specification(
        conversation_id=conversation_id,
        user_id=current_user.id,
    )
    return PlainTextResponse(
        content=data['markdown'],
        media_type='text/markdown; charset=utf-8',
    )
