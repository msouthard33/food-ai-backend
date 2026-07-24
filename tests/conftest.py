"""Fixtures for tests.

Sets APP_ENV=testing before importing the app so the production secret
validator does not fire and so tests use the local foodai_test database.
"""

import os

# Override env vars BEFORE any app module is imported.
# app/database.py creates the engine at module level; setting these here
# ensures the engine picks up the test DB on first import.
os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://matthewsouthard@localhost/foodai_test",
)
# Disable JWT secret so get_current_user falls back to UUID-as-token shortcut
os.environ.setdefault("SUPABASE_JWT_SECRET", "")
os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.config import get_settings

# Clear the lru_cache so Settings() re-reads from the env vars set above.
get_settings.cache_clear()

from app.database import Base, engine, async_session_factory  # noqa: E402
import app.models  # noqa: E402, F401 — register all ORM models for create_all
from app.main import app  # noqa: E402 — must be after env setup + cache clear

# Stable test user ID — reused across all authed tests
TEST_USER_ID = uuid.UUID("00000000-0000-4000-a000-000000000001")
TEST_USER_EMAIL = "testuser@foodai.test"

# Track whether tables have been created in this process
_tables_created = False


async def _ensure_tables() -> None:
    """Create all tables if not already done in this test process.

    Uses the app's own engine so connections stay on the same event loop.
    """
    global _tables_created
    if _tables_created:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _tables_created = True


@pytest.fixture
async def client():
    await _ensure_tables()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
async def authed_client():
    """HTTP client with a valid Authorization header for an existing test user.

    Creates tables (idempotent) and the test user so the dev-mode UUID auth
    shortcut in get_current_user can resolve it.
    """
    await _ensure_tables()

    async with async_session_factory() as session:
        await session.execute(
            text(
                "INSERT INTO users (id, email, timezone, onboarding_completed) "
                "VALUES (:id, :email, 'UTC', false) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"id": TEST_USER_ID, "email": TEST_USER_EMAIL},
        )
        await session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {TEST_USER_ID}"},
    ) as ac:
        yield ac
