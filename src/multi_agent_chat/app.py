import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.routing import Match

from multi_agent_chat.chatbot.controle.orquestrador import create_checkpointer
from multi_agent_chat.core.settings import Settings
from multi_agent_chat.routes import (
    agents,
    auth,
    chat,
    conversations,
    messages,
    users,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    async with create_checkpointer(settings.CHECKPOINT_DB_PATH) as saver:
        app.state.checkpointer = saver
        yield


app = FastAPI(lifespan=lifespan)

# Origens do front-end autorizadas (separadas por vírgula).
CORS_ORIGINS = [origin.strip() for origin in os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',') if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def _has_full_match(scope: dict) -> bool:
    return any(route.matches(scope)[0] == Match.FULL for route in app.router.routes)


@app.middleware('http')
async def accept_missing_trailing_slash(request: Request, call_next):
    """Aceita /rota e /rota/ sem o redirect 307 padrão do Starlette."""
    scope = request.scope
    path = scope['path']
    if not path.endswith('/') and not _has_full_match(scope):
        with_slash = {**scope, 'path': f'{path}/'}
        if _has_full_match(with_slash):
            scope['path'] = with_slash['path']
    return await call_next(request)


app.include_router(auth.router)
app.include_router(users.router)
app.include_router(agents.router)
app.include_router(conversations.router)

app.include_router(messages.router)
app.include_router(chat.router)
