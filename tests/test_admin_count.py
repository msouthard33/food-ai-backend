"""Admin foods count endpoint tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_admin_foods_count_with_key(client: AsyncClient):
    resp = await client.get(
        "/api/v1/admin/foods/count",
        headers={"X-Admin-Key": "test-admin-key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert isinstance(data["count"], int)
    assert data["count"] >= 0


@pytest.mark.asyncio
async def test_admin_foods_count_no_key(client: AsyncClient):
    resp = await client.get("/api/v1/admin/foods/count")
    assert resp.status_code == 422  # missing required header


@pytest.mark.asyncio
async def test_admin_foods_count_wrong_key(client: AsyncClient):
    resp = await client.get(
        "/api/v1/admin/foods/count",
        headers={"X-Admin-Key": "wrong-key"},
    )
    assert resp.status_code == 403
