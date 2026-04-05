"""Fixtures for tests."""

import pytest
from httpx import ASyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import Connectwdocs
from app.main import app


@pytest.fixture
async def client():
    async with AsyncClient(app=app) as ac:
        yield ac


@pytest.fixture
async def db():
    engine = create_async_engine()
    yield engine
