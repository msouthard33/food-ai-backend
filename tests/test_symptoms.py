"""Symptom endpoint tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_log_symptom(client: AsyncClient):
    resp = await client.post(
        "/api/v1/symptoms/log",
        json={
            "timestamp": "2026-04-04T12:30:00Z",
            "symptom_type": "nausea",
            "vas_score": 5,
            "notes": "Very mild, after eating shoeblas",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["symptom_type"] == "nausea"
    assert data["vas_score"] == 5


@pytest.mark.asyncio
async def test_list_symptoms(client: AsyncClient):
    resp = await client.get("/api/v1/symptoms")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
