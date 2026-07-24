"""W2-1 Pillar 1 — semantic search, meal decomposition, allergen inference.

These tests exercise the shipped/offline paths only: the embedding sidecar
tables are not populated in the test DB, so semantic search degrades to lexical
fallback, and meal decomposition uses the deterministic heuristic splitter (no
live LLM). This is exactly the behavior that ships until pgvector is activated.
"""

import uuid
from decimal import Decimal

import pytest
from httpx import AsyncClient

from app.database import async_session_factory
from app.models.enums import ComponentType
from app.models.food import FoodComponentDetail, FoodEntry
from app.services import allergen_inference, meal_decomposition
from app.services.llm_provider import EMBEDDING_DIM, embed_text

# ---------------------------------------------------------------------------
# Unit tests — pure functions, no DB
# ---------------------------------------------------------------------------


def test_heuristic_split_multi_ingredient():
    parts = meal_decomposition._heuristic_split("chicken with rice and broccoli")
    ingredients = [p[0] for p in parts]
    assert ingredients == ["chicken", "rice", "broccoli"]


def test_heuristic_split_portion_extraction():
    parts = meal_decomposition._heuristic_split("2 cups spinach")
    assert parts[0][0] == "spinach"
    assert parts[0][1] is not None  # portion captured


def test_embed_offline_is_deterministic_and_sized():
    a = embed_text("sourdough bread", provider="offline")
    b = embed_text("sourdough bread", provider="offline")
    assert a == b  # stable across calls/processes (hashlib, not builtin hash)
    assert len(a) == EMBEDDING_DIM
    assert embed_text("apple", provider="offline") != a


def test_allergen_flag_from_kb_component_high_level():
    food = FoodEntry(id=uuid.uuid4(), name="Aged Cheddar")
    food.components = [
        FoodComponentDetail(component_type=ComponentType.HISTAMINES, level=Decimal("3.0"))
    ]
    flag = allergen_inference.flag_from_kb_component(food, "histamines")
    assert flag is not None
    assert flag.level_label == "high"
    assert flag.provenance == "kb"
    assert 0.0 <= flag.confidence <= 1.0
    assert flag.caveat  # non-empty, UI-ready


def test_allergen_flag_returns_none_when_component_absent():
    food = FoodEntry(id=uuid.uuid4(), name="Plain Rice")
    food.components = [
        FoodComponentDetail(component_type=ComponentType.FODMAP, level=Decimal("0.5"))
    ]
    assert allergen_inference.flag_from_kb_component(food, "histamines") is None


# ---------------------------------------------------------------------------
# Integration tests — through the API (offline / lexical-fallback paths)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_search_endpoint_falls_back_cleanly(authed_client: AsyncClient):
    """No embeddings in test DB → lexical fallback, never a 500."""
    resp = await authed_client.get("/api/v1/foods/search/semantic?q=bread")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"query", "total", "items"}
    for item in data["items"]:
        assert item["source"] == "lexical_fallback"


@pytest.mark.asyncio
async def test_semantic_search_finds_seeded_food(authed_client: AsyncClient):
    """Seed a distinctively-named food; lexical fallback must surface it."""
    unique = "Zzxq Sourdough Loaf"
    async with async_session_factory() as session:
        session.add(FoodEntry(id=uuid.uuid4(), name=unique, category="grains"))
        await session.commit()

    resp = await authed_client.get("/api/v1/foods/search/semantic?q=Zzxq")
    assert resp.status_code == 200
    names = [i["name"] for i in resp.json()["items"]]
    assert unique in names


@pytest.mark.asyncio
async def test_decompose_endpoint_heuristic(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/foods/decompose",
        json={"meal_text": "chicken with rice and broccoli"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["structured"] is True
    assert data["source"] == "heuristic"  # no live LLM in tests
    assert len(data["ingredients"]) == 3
    for ing in data["ingredients"]:
        assert "confidence" in ing and 0.0 <= ing["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_decompose_endpoint_empty_input(authed_client: AsyncClient):
    resp = await authed_client.post("/api/v1/foods/decompose", json={"meal_text": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert data["structured"] is False
    assert data["ingredients"] == []


@pytest.mark.asyncio
async def test_semantic_and_decompose_require_auth(client: AsyncClient):
    """Both new endpoints sit behind get_current_user like the rest of /foods."""
    # main's auth returns 422 when the required auth header is absent (same as
    # the rest of the API); the point is that anonymous access is rejected.
    r1 = await client.get("/api/v1/foods/search/semantic?q=bread")
    r2 = await client.post("/api/v1/foods/decompose", json={"meal_text": "toast"})
    assert r1.status_code in (401, 403, 422)
    assert r2.status_code in (401, 403, 422)
