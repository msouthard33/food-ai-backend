"""Food database endpoint tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_food(client: AsyncClient):
    resp = await client.get("/api/v1/foods?name=bread")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
