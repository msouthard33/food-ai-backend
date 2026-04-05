"""Meal endpoint tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_meal(client: AsyncClient):
    resp = await client.post(
        "/api/v1/meals",
        json={
            "timestamp": "2026-04-04T12:30:00Z",
            "meal_type": "lunch",
            "raw_description": "Grilled chicken salad with avocado",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["meal_type"] == "lunch"
    assert data["raw_description"] == "Grilled chicken salad with avocado"
    assert data["processing_status"] == "pending"
    assert "id" in data
