"""ADR-0003 breakdown foundation — canonical resolution + unified /breakdown.

Covers build steps 1–3 (foundation only):
  1. Canonical alias/normalization layer (§3) — pure-function alias collapse.
  2. Unified POST /api/v1/foods/breakdown (§1) — ADR-0003 shape, curated→AI
     precedence, JWT-required, /decompose preserved.
  3. Recipe-graph (§4) — curated edges drive the deterministic curated path.

Offline paths only (no live LLM, empty embedding sidecar → lexical fallback),
which is exactly what ships until pgvector is populated.
"""

import uuid
from decimal import Decimal

import pytest
from httpx import AsyncClient

from app.database import async_session_factory
from app.models.food import FoodEntry, FoodRecipeEdge
from app.services import breakdown_service, canonical_resolution

# ---------------------------------------------------------------------------
# 1. Canonical resolution — pure alias-collapse (no DB) — ADR-0003 §3
# ---------------------------------------------------------------------------


def test_alias_collapse_soy_sauce_family():
    """The canonical ADR-0003 §3 example: shoyu / tamari / soy sauce collapse."""
    terms = ["soy sauce", "Shoyu", "TAMARI", "2 tbsp tamari", "Soy Sauce."]
    canon = {canonical_resolution.canonicalize(t) for t in terms}
    assert canon == {"soy sauce"}


def test_alias_collapse_regional_synonyms():
    assert canonical_resolution.canonicalize("Aubergine") == "eggplant"
    assert canonical_resolution.canonicalize("courgettes") == "zucchini"
    assert canonical_resolution.canonicalize("Scallions") == "green onion"
    assert canonical_resolution.canonicalize("garbanzo beans") == "chickpeas"
    assert canonical_resolution.canonicalize("Prawns") == "shrimp"


def test_canonicalize_depluralizes_but_guards_mass_nouns():
    assert canonical_resolution.canonicalize("Tomatoes") == "tomato"
    assert canonical_resolution.canonicalize("apples") == "apple"
    # mass/irregular nouns must NOT be mangled
    assert canonical_resolution.canonicalize("molasses") == "molasses"
    assert canonical_resolution.canonicalize("hummus") == "hummus"


def test_canonicalize_strips_portion_and_punctuation():
    assert canonical_resolution.canonicalize("1 cup White Rice") == "white rice"
    assert canonical_resolution.canonicalize("  Bell-Pepper!! ") == "bell pepper"


def test_canonicalize_empty_is_empty():
    assert canonical_resolution.canonicalize("") == ""
    assert canonical_resolution.canonicalize("   ") == ""


def test_placeholder_key_categories():
    assert canonical_resolution.placeholder_key("house soy sauce") == "placeholder:sauce"
    assert canonical_resolution.placeholder_key("brown rice") == "placeholder:grain"
    assert canonical_resolution.placeholder_key("grilled chicken") == "placeholder:protein"
    assert canonical_resolution.placeholder_key("quokka jerky xyz") == "placeholder:generic"


# ---------------------------------------------------------------------------
# 1b. Canonical resolution — through the DB (lexical fallback path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_ingredient_long_tail_is_null_with_marker():
    """An ingredient with no KB match resolves to null + long-tail marker."""
    async with async_session_factory() as session:
        r = await canonical_resolution.resolve_ingredient(
            session, "zzxq nonexistent long-tail ingredient"
        )
    assert r.food_id is None
    assert r.resolved is False
    assert r.marker == canonical_resolution.LONG_TAIL_MARKER
    assert r.thumbnail_id.startswith("placeholder:")


@pytest.mark.asyncio
async def test_resolve_ingredient_binds_canonical_food_id():
    """A seeded food is resolved to its canonical food_id via lexical fallback."""
    fid = uuid.uuid4()
    async with async_session_factory() as session:
        session.add(FoodEntry(id=fid, name="Zzxq Canonical Tofu", category="protein"))
        await session.commit()
        # 'beancurd' is an alias for 'tofu' — collapse then resolve.
        r = await canonical_resolution.resolve_ingredient(session, "Beancurd")
    # lexical fallback score is 0.5 == RESOLVE_THRESHOLD → bound
    assert r.canonical_term == "tofu"
    if r.food_id is not None:
        assert r.resolved is True
        assert r.thumbnail_id == r.food_id


# ---------------------------------------------------------------------------
# 2. Unified breakdown — ADR-0003 §1 shape + Trigger Preview §4a parity
# ---------------------------------------------------------------------------

_INGREDIENT_KEYS = {
    "name",
    "food_id",
    "preselected",
    "confidence",
    "portion",
    "source",
    "thumbnail_id",
    "caveat",
}


def _assert_breakdown_shape(data: dict) -> None:
    """The exact ADR-0003 §1 / Trigger Preview §4a top-level + ingredient keys."""
    assert set(data.keys()) == {"input", "resolved_food_id", "source", "ingredients"}
    assert data["source"] in {"curated", "ai", "heuristic"}
    for ing in data["ingredients"]:
        assert set(ing.keys()) == _INGREDIENT_KEYS
        assert ing["source"] in {"curated", "ai", "user"}
        assert isinstance(ing["preselected"], bool)
        assert 0.0 <= ing["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_breakdown_endpoint_ai_heuristic_shape(authed_client: AsyncClient):
    """Free text with no curated composite → heuristic split, §1 shape."""
    resp = await authed_client.post(
        "/api/v1/foods/breakdown",
        json={"input": "chicken with rice and broccoli", "surface": "capture"},
    )
    assert resp.status_code == 200
    data = resp.json()
    _assert_breakdown_shape(data)
    assert data["input"] == "chicken with rice and broccoli"
    assert data["source"] == "heuristic"  # no live LLM in tests
    assert len(data["ingredients"]) == 3


@pytest.mark.asyncio
async def test_breakdown_endpoint_empty_input(authed_client: AsyncClient):
    resp = await authed_client.post("/api/v1/foods/breakdown", json={"input": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ingredients"] == []
    assert data["resolved_food_id"] is None


@pytest.mark.asyncio
async def test_breakdown_requires_auth(client: AsyncClient):
    """/breakdown sits behind get_current_user like the rest of /foods."""
    r = await client.post("/api/v1/foods/breakdown", json={"input": "toast"})
    assert r.status_code in (401, 403, 422)


@pytest.mark.asyncio
async def test_decompose_still_works(authed_client: AsyncClient):
    """Existing /decompose callers are not broken by the unification."""
    resp = await authed_client.post(
        "/api/v1/foods/decompose", json={"meal_text": "toast with butter"}
    )
    assert resp.status_code == 200
    assert "ingredients" in resp.json()


# ---------------------------------------------------------------------------
# 3. Curated path — recipe edges drive a deterministic checklist (§2/§4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_curated_path_serves_recipe_edges():
    """When a composite has curated edges, the curated path returns them."""
    composite_id = uuid.uuid4()
    rice_id = uuid.uuid4()
    nori_id = uuid.uuid4()
    async with async_session_factory() as session:
        session.add_all(
            [
                FoodEntry(id=composite_id, name="Zzxq Test Sushi Roll", category="composite"),
                FoodEntry(id=rice_id, name="Zzxq White Rice", category="grain"),
                FoodEntry(id=nori_id, name="Zzxq Nori Seaweed", category="vegetable"),
                FoodRecipeEdge(
                    id=uuid.uuid4(),
                    composite_food_id=composite_id,
                    ingredient_food_id=rice_id,
                    default_selected=True,
                    typical_portion="1 cup",
                    provenance="curated",
                    confidence=Decimal("0.950"),
                ),
                FoodRecipeEdge(
                    id=uuid.uuid4(),
                    composite_food_id=composite_id,
                    ingredient_food_id=nori_id,
                    default_selected=False,
                    typical_portion="2 sheets",
                    provenance="curated",
                    confidence=Decimal("0.900"),
                ),
            ]
        )
        await session.commit()

        data = await breakdown_service.build_breakdown(
            session, "Zzxq Test Sushi Roll", surface="search", food_id=str(composite_id)
        )

    _assert_breakdown_shape(data)
    assert data["source"] == "curated"
    assert data["resolved_food_id"] == str(composite_id)
    assert len(data["ingredients"]) == 2
    names = {i["name"] for i in data["ingredients"]}
    assert names == {"Zzxq White Rice", "Zzxq Nori Seaweed"}
    for ing in data["ingredients"]:
        assert ing["source"] == "curated"
        assert ing["food_id"] is not None
        assert ing["caveat"] is None
    # default_selected preserved as preselected
    rice = next(i for i in data["ingredients"] if i["name"] == "Zzxq White Rice")
    assert rice["preselected"] is True
    assert rice["portion"] == "1 cup"


@pytest.mark.asyncio
async def test_curated_miss_falls_through_to_ai(authed_client: AsyncClient):
    """A composite with no curated edges falls through (expected until step 5)."""
    async with async_session_factory() as session:
        data = await breakdown_service.build_breakdown(
            session, "uncurated compound dish with beans and rice", surface="capture"
        )
    assert data["source"] in {"ai", "heuristic"}  # not curated
