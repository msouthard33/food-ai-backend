"""Meal endpoint tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_meal(authed_client: AsyncClient):
    resp = await authed_client.post(
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


@pytest.mark.asyncio
async def test_get_meal(authed_client: AsyncClient):
    # Create a meal first
    create_resp = await authed_client.post(
        "/api/v1/meals",
        json={
            "timestamp": "2026-04-04T18:00:00Z",
            "meal_type": "dinner",
            "raw_description": "Pasta with tomato sauce",
        },
    )
    assert create_resp.status_code == 201
    meal_id = create_resp.json()["id"]

    # Fetch it
    resp = await authed_client.get(f"/api/v1/meals/{meal_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == meal_id
    assert data["meal_type"] == "dinner"
    assert data["raw_description"] == "Pasta with tomato sauce"


@pytest.mark.asyncio
async def test_get_meal_not_found(authed_client: AsyncClient):
    resp = await authed_client.get("/api/v1/meals/00000000-0000-4000-a000-000000000099")
    assert resp.status_code == 404
