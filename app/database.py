"""Async SQLAlchemy engine and session factory."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config import get_settings

settings = get_settings()

# Use NullPool when connecting through Supabase's PgBouncer pooler (port 6543)
# to avoid double-pooling. Also use NullPool in testing to avoid event-loop
# conflicts across tests (each async test may get a new event loop).
_is_pooler = ":6543/" in settings.database_url
_use_null_pool = _is_pooler or settings.is_testing

engine = create_async_engine(
    settings.database_url,
    echo=(not settings.is_production),
    **({
        "poolclass": NullPool,
    } if _use_null_pool else {
        "pool_size": 5,
        "max_overflow": 10,
        "pool_pre_ping": True,
    }),
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
