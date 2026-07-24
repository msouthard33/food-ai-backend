"""Barcode endpoint tests (W2-3 box 5).

The live Open Food Facts call (`barcode_service.fetch_off_product`) is mocked in
every test — the suite never touches the network.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import AsyncClient

# Import the 50-item benchmark set + the OFF-payload synthesiser.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from w2_3_barcode_benchmark import BARCODE_TEST_SET, synthetic_off_product  # noqa: E402

_TIER_VOCAB = {"Well-established", "Some evidence", "AI estimate"}

_PATCH_TARGET = "app.services.barcode_service.fetch_off_product"


def _assert_tier_everywhere(data: dict) -> None:
    """Every barcode response must carry a tier_label at product + ingredient level."""
    assert data["tier_label"] in _TIER_VOCAB
    for ing in data["ingredients"]:
        assert ing["tier_label"] in _TIER_VOCAB


@pytest.mark.asyncio
async def test_barcode_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/barcode/0049000000443")
    assert resp.status_code == 422  # missing Authorization header


@pytest.mark.asyncio
async def test_barcode_matched(authed_client: AsyncClient):
    product = synthetic_off_product(
        {"product_name": "Kraft Macaroni and Cheese",
         "ingredients": ["wheat", "cheddar cheese", "milk", "whey", "butter"]}
    )
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=product)):
        resp = await authed_client.get("/api/v1/barcode/021000658830")
    assert resp.status_code == 200
    data = resp.json()
    assert data["off_found"] is True
    assert data["status"] == "matched"
    assert data["source"] == "openfoodfacts+kb"
    assert data["matched_count"] >= 1
    assert 0.0 <= data["confidence"] <= 1.0
    _assert_tier_everywhere(data)


@pytest.mark.asyncio
async def test_barcode_not_found(authed_client: AsyncClient):
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=None)):
        resp = await authed_client.get("/api/v1/barcode/000000000000")
    assert resp.status_code == 200
    data = resp.json()
    assert data["off_found"] is False
    assert data["status"] == "not_found"
    assert data["ingredients"] == []
    assert data["tier_label"] in _TIER_VOCAB


@pytest.mark.asyncio
async def test_barcode_off_unavailable(authed_client: AsyncClient):
    """A network/HTTP failure degrades gracefully to a tier-labelled response."""
    boom = AsyncMock(side_effect=httpx.ConnectError("off down"))
    with patch(_PATCH_TARGET, new=boom):
        resp = await authed_client.get("/api/v1/barcode/021000658830")
    assert resp.status_code == 200
    data = resp.json()
    assert data["off_found"] is False
    assert data["status"] == "not_found"
    assert "unavailable" in (data["message"] or "").lower()
    assert data["tier_label"] in _TIER_VOCAB


@pytest.mark.asyncio
async def test_barcode_ai_fallback_when_no_kb_match(authed_client: AsyncClient):
    """OFF has the product but nothing maps to the KB -> AI-decomposition source."""
    product = synthetic_off_product(
        {"product_name": "Zzxq Blorptangle 9000", "ingredients": ["qwertium extract"]}
    )
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=product)):
        resp = await authed_client.get("/api/v1/barcode/111111111111")
    assert resp.status_code == 200
    data = resp.json()
    assert data["off_found"] is True
    assert data["status"] == "off_only"
    assert data["source"] == "openfoodfacts+ai_decomposition"
    assert data["matched_count"] == 0
    _assert_tier_everywhere(data)


@pytest.mark.asyncio
async def test_barcode_e2e_50_grocery_items(authed_client: AsyncClient):
    """Box 5: structured trigger profile E2E for the 50-item grocery set."""
    assert len(BARCODE_TEST_SET) == 50
    by_barcode = {e["barcode"]: synthetic_off_product(e) for e in BARCODE_TEST_SET}

    async def fake_fetch(barcode: str):
        return by_barcode.get(barcode)

    matched_items = 0
    with patch(_PATCH_TARGET, new=AsyncMock(side_effect=fake_fetch)):
        for entry in BARCODE_TEST_SET:
            resp = await authed_client.get(f"/api/v1/barcode/{entry['barcode']}")
            assert resp.status_code == 200, entry["product_name"]
            data = resp.json()
            # Every item returns a structured profile with a product-level tier.
            assert data["off_found"] is True
            assert "ingredients" in data
            _assert_tier_everywhere(data)
            if data["matched_count"] > 0:
                matched_items += 1

    # The pipeline should resolve the majority of common grocery items to the KB.
    assert matched_items >= 30, f"only {matched_items}/50 items matched the KB"
