"""Barcode endpoint tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_barcode_stub(authed_client: AsyncClient):
    resp = await authed_client.get("/api/v1/barcode/0049000000443")
    assert resp.status_code == 200
    data = resp.json()
    assert data["barcode"] == "0049000000443"
    assert data["status"] == "not_implemented"


@pytest.mark.asyncio
async def test_barcode_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/barcode/0049000000443")
    assert resp.status_code == 422  # missing Authorization header
