"""Food database endpoint tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_food(authed_client: AsyncClient):
    resp = await authed_client.get("/api/v1/foods/search?q=bread")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
