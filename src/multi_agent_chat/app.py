from contextlib import asynccontextmanager

from fastapi import FastAPI

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

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(agents.router)
app.include_router(conversations.router)

app.include_router(messages.router)
app.include_router(chat.router)
