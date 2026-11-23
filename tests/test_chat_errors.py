import json
from http import HTTPStatus
from unittest.mock import MagicMock

import pytest
from langchain_google_genai.chat_models import (
    GoogleAPIError,
    GoogleAuthenticationError,
    GoogleInvalidRequestError,
    GoogleModelNotFoundError,
    GoogleRateLimitError,
)

from multi_agent_chat.app import app
from multi_agent_chat.chatbot.controle.orquestrador import Orquestrador
from multi_agent_chat.core.chatbot import get_model_factory
from multi_agent_chat.services.chat import gemini_error_info

REJECTED = 'Gemini API rejected the request. Check the API key and the model settings.'


@pytest.mark.parametrize(
    ('exc', 'status', 'detail'),
    [
        (
            GoogleRateLimitError('quota'),
            HTTPStatus.TOO_MANY_REQUESTS,
            'Gemini API rate limit exceeded. Try again later.',
        ),
        (
            GoogleAPIError(503, {}),
            HTTPStatus.SERVICE_UNAVAILABLE,
            'Gemini API is temporarily unavailable.',
        ),
        (
            GoogleAPIError(500, {}),
            HTTPStatus.BAD_GATEWAY,
            'Error communicating with Gemini API.',
        ),
        (GoogleInvalidRequestError('API key not valid'), 502, REJECTED),
        (GoogleAuthenticationError('bad key'), 502, REJECTED),
        (GoogleModelNotFoundError('no model'), 502, REJECTED),
    ],
)
def test_gemini_error_info(exc, status, detail):
    assert gemini_error_info(exc) == (status, detail)


@pytest.fixture
def chat_url(conversation):
    return f'/conversations/{conversation.id}/chat'


@pytest.fixture
def headers(token):
    return {'Authorization': f'Bearer {token}'}


@pytest.fixture(autouse=True)
def fake_model_factory(client):
    app.dependency_overrides[get_model_factory] = lambda: lambda agent: MagicMock()


@pytest.fixture
def gemini_rejects(monkeypatch):
    error = GoogleInvalidRequestError('API key not valid. SECRET-DETAIL')

    async def iniciar(self, thread_id, descricao):
        raise error

    async def stream(self, *args, **kwargs):
        raise error
        yield  # pragma: no cover

    monkeypatch.setattr(Orquestrador, 'iniciar', iniciar)
    monkeypatch.setattr(Orquestrador, 'stream', stream)


def test_chat_returns_502_when_gemini_rejects(client, chat_url, headers, gemini_rejects):
    response = client.post(chat_url, headers=headers, json={'content': 'oi'})
    assert response.status_code == HTTPStatus.BAD_GATEWAY
    assert response.json()['detail'] == REJECTED


def test_stream_sends_error_event_when_gemini_rejects(client, chat_url, headers, gemini_rejects):
    response = client.post(f'{chat_url}/stream', headers=headers, json={'content': 'oi'})
    assert response.status_code == HTTPStatus.OK
    text = response.text
    assert 'event: error' in text
    assert 'SECRET-DETAIL' not in text  # não vaza o texto cru da exceção
    data_line = next(line for line in text.splitlines() if line.startswith('data:') and 'detail' in line)
    payload = json.loads(data_line.removeprefix('data:').strip())
    assert payload == {
        'detail': REJECTED,
        'status_code': HTTPStatus.BAD_GATEWAY,
    }
    assert 'event: done' not in text
