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

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import get_settings

# Clear the lru_cache so Settings() re-reads from the env vars set above.
get_settings.cache_clear()

from app.main import app  # noqa: E402 — must be after env setup + cache clear


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
