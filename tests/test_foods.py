"""Food database endpoint tests."""

import uuid

import pytest
from httpx import AsyncClient

from app.database import async_session_factory
from app.models.food import FoodEntry


@pytest.mark.asyncio
async def test_get_food(authed_client: AsyncClient):
    resp = await authed_client.get("/api/v1/foods/search?q=bread")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data


@pytest.mark.asyncio
async def test_search_tolerates_null_common_names(authed_client: AsyncClient):
    """>half the KB has no common_names (NULL). Search must not 500 on those."""
    name = "Zqx Nullnames Loaf"
    async with async_session_factory() as s:
        s.add(FoodEntry(id=uuid.uuid4(), name=name, category="test", common_names=None))
        await s.commit()

    resp = await authed_client.get("/api/v1/foods/search?q=Zqx")
    assert resp.status_code == 200
    match = [i for i in resp.json()["items"] if i["name"] == name]
    assert match, "food with NULL common_names should appear in results"
    assert match[0]["common_names"] == []  # coerced, not null
