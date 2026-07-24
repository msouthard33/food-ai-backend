"""Meal endpoint tests."""

import base64
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

# A minimal valid JPEG header so PhotoAnalyzeRequest validation passes without a
# real photo. The vision call itself is always mocked (no live API, no PHI bytes).
_FAKE_JPEG = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 512).decode()

_TIER_VOCAB = {"Well-established", "Some evidence", "AI estimate"}

_VISION_PATCH = "app.services.llm_provider.LLMProvider.analyze_meal_photo"


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


# ---------------------------------------------------------------------------
# Photo analysis (W2-3 box 6). The vision LLM call is always mocked.
# ---------------------------------------------------------------------------

def _vision_stub(foods: list[dict]) -> AsyncMock:
    """Build an AsyncMock replacing LLMProvider.analyze_meal_photo."""
    return AsyncMock(return_value=foods)


@pytest.mark.asyncio
async def test_analyze_photo_requires_auth(client: AsyncClient):
    resp = await client.post("/api/v1/meals/analyze-photo", json={"image_base64": _FAKE_JPEG})
    assert resp.status_code in (401, 422)  # missing Authorization header


@pytest.mark.asyncio
async def test_analyze_photo_rejects_non_image(authed_client: AsyncClient):
    bad = base64.b64encode(b"this is definitely not an image payload").decode()
    resp = await authed_client.post("/api/v1/meals/analyze-photo", json={"image_base64": bad})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_analyze_photo_success(authed_client: AsyncClient):
    foods = [
        {"food": "grilled chicken breast", "portion": "6 oz", "confidence": 0.92,
         "tier_label": "Well-established"},
        {"food": "white rice", "portion": "1 cup", "confidence": 0.6,
         "tier_label": "Some evidence"},
        {"food": "mystery garnish", "portion": "1 tsp", "confidence": 0.3,
         "tier_label": "AI estimate"},
    ]
    with patch(_VISION_PATCH, new=_vision_stub(foods)):
        resp = await authed_client.post(
            "/api/v1/meals/analyze-photo", json={"image_base64": _FAKE_JPEG}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["food_count"] == 3
    assert len(data["foods"]) == 3
    assert data["photo_analysis_model"]
    # Box-6 / D9 rule: EVERY food carries a tier_label from the D9 vocabulary.
    for food in data["foods"]:
        assert food["tier_label"] in _TIER_VOCAB
        assert 0.0 <= food["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_analyze_photo_vision_unavailable_still_tier_labelled(authed_client: AsyncClient):
    fallback = [{
        "food": "Photo analysis unavailable", "portion": "N/A", "confidence": 0.0,
        "tier_label": "AI estimate", "error": "vision_provider_unavailable",
    }]
    with patch(_VISION_PATCH, new=_vision_stub(fallback)):
        resp = await authed_client.post(
            "/api/v1/meals/analyze-photo", json={"image_base64": _FAKE_JPEG}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["food_count"] == 0
    assert data["foods"][0]["tier_label"] in _TIER_VOCAB
    assert data["foods"][0]["error"] == "vision_provider_unavailable"


@pytest.mark.asyncio
async def test_analyze_photo_30_meal_set(authed_client: AsyncClient):
    """Box 6: structured output for a 30-meal set, each carrying a tier_label."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from w2_3_photo_benchmark import MEAL_TEST_SET  # noqa: E402

    assert len(MEAL_TEST_SET) == 30

    for meal in MEAL_TEST_SET:
        # Synthesise the vision output the model would return for this meal.
        stub_foods = [
            {"food": name, "portion": "1 serving", "confidence": 0.8}
            for name in meal["expected_foods"]
        ]
        with patch(_VISION_PATCH, new=_vision_stub(stub_foods)):
            resp = await authed_client.post(
                "/api/v1/meals/analyze-photo", json={"image_base64": _FAKE_JPEG}
            )
        assert resp.status_code == 200, meal["description"]
        data = resp.json()
        assert data["food_count"] >= meal["min_foods"]
        for food in data["foods"]:
            assert food["tier_label"] in _TIER_VOCAB
