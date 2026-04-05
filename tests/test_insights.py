"""Insights/trigger endpoint tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_triggers_empty(client: AsyncClient):
    resp = await client.get("/api/v1/insights/triggers")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["triggers"] == []
