from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from multi_agent_chat.core.settings import Settings

engine = create_async_engine(Settings().DATABASE_URL)


# Garante que que o SQLite respeite as restrições de chave estrangeira
@event.listens_for(engine.sync_engine, 'connect')
def set_sqlite_pragma(dbapi_connection, _connection_record):
    if engine.dialect.name == 'sqlite':
        cursor = dbapi_connection.cursor()
        cursor.execute('PRAGMA foreign_keys=ON')
        cursor.close()


async def get_session():
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
