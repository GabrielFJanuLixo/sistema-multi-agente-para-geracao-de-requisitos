from collections.abc import Callable
from http import HTTPStatus
from typing import Any

from fastapi import HTTPException, Request
from langchain_google_genai import ChatGoogleGenerativeAI

from multi_agent_chat.core.settings import Settings
from multi_agent_chat.models.agent import Agent

ModelFactory = Callable[[Agent], Any]


def build_gemini_model(agent: Agent) -> ChatGoogleGenerativeAI:
    api_key = Settings().GOOGLE_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail='GOOGLE_API_KEY is not configured',
        )
    return ChatGoogleGenerativeAI(
        model=agent.model,
        temperature=agent.temperature,
        google_api_key=api_key,
        timeout=120,
        max_retries=6,
    )


def get_model_factory() -> ModelFactory:
    return build_gemini_model


def get_checkpointer(request: Request) -> Any:
    return getattr(request.app.state, 'checkpointer', None)
