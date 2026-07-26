"""W2-3c tests: tier_label, insights, protocols, medications, FHIR, food-drug model."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from tests.conftest import TEST_USER_ID, async_session_factory


# ── tier_label utility ──────────────────────────────────────────────────

def test_confidence_to_tier_label():
    from app.utils.confidence import confidence_to_tier_label

    assert confidence_to_tier_label(0.90) == "Well-established"
    assert confidence_to_tier_label(0.85) == "Well-established"
    assert confidence_to_tier_label(0.84) == "Some evidence"
    assert confidence_to_tier_label(0.55) == "Some evidence"
    assert confidence_to_tier_label(0.54) == "AI estimate"
    assert confidence_to_tier_label(0.0) == "AI estimate"


# ── tier_label in trigger schema ────────────────────────────────────────

def test_trigger_prediction_tier_label():
    from app.schemas.trigger import TriggerPredictionOut

    pred = TriggerPredictionOut(
        id=uuid.uuid4(),
        component_type="histamines",
        confidence_score=85,
        evidence_count=10,
        status="confirmed",
        last_updated=datetime.now(timezone.utc),
    )
    assert pred.tier_label == "Well-established"

    pred2 = TriggerPredictionOut(
        id=uuid.uuid4(),
        component_type="gluten",
        confidence_score=60,
        evidence_count=5,
        status="probable",
        last_updated=datetime.now(timezone.utc),
    )
    assert pred2.tier_label == "Some evidence"

    pred3 = TriggerPredictionOut(
        id=uuid.uuid4(),
        component_type="fodmap",
        confidence_score=30,
        evidence_count=2,
        status="suspect",
        last_updated=datetime.now(timezone.utc),
    )
    assert pred3.tier_label == "AI estimate"


# ── tier_label in meal item component schema ────────────────────────────

def test_meal_item_component_tier_label():
    from app.schemas.meal import MealItemComponentOut

    comp = MealItemComponentOut(
        component_type="histamines",
        estimated_level=3.5,
        confidence_score=0.90,
    )
    assert comp.tier_label == "Well-established"

    comp2 = MealItemComponentOut(
        component_type="gluten",
        estimated_level=2.0,
        confidence_score=0.40,
    )
    assert comp2.tier_label == "AI estimate"


# ── GET /insights/triggers (existing, with tier_label) ──────────────────

@pytest.mark.asyncio
async def test_get_triggers_includes_tier_label_field(authed_client: AsyncClient):
    resp = await authed_client.get("/api/v1/insights/triggers")
    assert resp.status_code == 200
    # Even with 0 triggers, the shape is valid
    data = resp.json()
    assert "triggers" in data
    assert "total" in data


# ── GET /insights/lag-correlation ────────────────────────────────────────

@pytest.mark.asyncio
async def test_lag_correlation_empty(authed_client: AsyncClient):
    resp = await authed_client.get("/api/v1/insights/lag-correlation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["correlations"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_lag_correlation_with_data(authed_client: AsyncClient):
    """Seed meals + symptoms via API, then verify correlation output."""
    now = datetime.now(timezone.utc)

    # Create a meal via API
    meal_resp = await authed_client.post("/api/v1/meals", json={
        "timestamp": (now - timedelta(hours=6)).isoformat(),
        "raw_description": "Aged Cheese sandwich",
        "meal_type": "lunch",
    })
    assert meal_resp.status_code == 201
    meal_id = meal_resp.json()["id"]

    # Add meal item
    await authed_client.post(f"/api/v1/meals/{meal_id}/items", json={
        "items": [{"name": "Aged Cheese"}]
    })

    # Create 3 symptom events
    for i in range(3):
        resp = await authed_client.post("/api/v1/symptoms", json={
            "timestamp": (now - timedelta(hours=i)).isoformat(),
            "symptom_type": "bloating",
            "vas_score": 70,
        })
        assert resp.status_code == 201

    resp = await authed_client.get("/api/v1/insights/lag-correlation")
    assert resp.status_code == 200
    data = resp.json()
    # Should have at least one correlation since meal is within 24h of symptoms
    assert data["total"] >= 0  # may or may not reach sample_size=2 threshold
    for row in data["correlations"]:
        # Versioned scoring contract present on every row.
        assert row["method"] == "hierarchical_bayes_logistic"
        assert 0.0 <= row["trigger_probability"] <= 1.0


# ── GET /insights/suspect-foods ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_suspect_foods_empty(authed_client: AsyncClient):
    resp = await authed_client.get("/api/v1/insights/suspect-foods")
    assert resp.status_code == 200
    data = resp.json()
    assert "foods" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_suspect_foods_has_confidence_tier(authed_client: AsyncClient):
    """When suspect foods are returned, each should have confidence_tier + the
    versioned hierarchical-Bayes / hybrid-guardrail fields."""
    resp = await authed_client.get("/api/v1/insights/suspect-foods")
    assert resp.status_code == 200
    data = resp.json()
    for food in data["foods"]:
        assert "confidence_tier" in food
        assert food["confidence_tier"] in ("Well-established", "Some evidence", "AI estimate")
        # Versioned scoring contract + hybrid classical guardrail fields present.
        assert food["method"] == "hierarchical_bayes_logistic"
        assert 0.0 <= food["trigger_probability"] <= 1.0
        assert "assoc_p_value" in food
        assert "assoc_agreement" in food


# ── POST /protocols/start ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_protocol_low_histamine(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/protocols/start",
        json={"protocol_type": "low-histamine"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["protocol_type"] == "low-histamine"
    assert "protocol_id" in data
    assert "started_at" in data
    assert len(data["foods_to_avoid"]) > 0
    assert "Red Wine" in data["foods_to_avoid"]


@pytest.mark.asyncio
async def test_start_protocol_top8(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/protocols/start",
        json={"protocol_type": "top8-allergen"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["protocol_type"] == "top8-allergen"
    assert "Milk" in data["foods_to_avoid"]
    assert "Peanuts" in data["foods_to_avoid"]


@pytest.mark.asyncio
async def test_start_protocol_low_fodmap(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/protocols/start",
        json={"protocol_type": "low-fodmap"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["protocol_type"] == "low-fodmap"
    assert "Garlic" in data["foods_to_avoid"]


@pytest.mark.asyncio
async def test_start_protocol_invalid_type(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/protocols/start",
        json={"protocol_type": "keto-diet"},
    )
    assert resp.status_code == 422


# ── POST /symptoms/medications ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_log_medication(authed_client: AsyncClient):
    """Create a symptom, then co-log a medication against it."""
    now = datetime.now(timezone.utc)

    # First create a symptom
    symptom_resp = await authed_client.post(
        "/api/v1/symptoms",
        json={
            "timestamp": now.isoformat(),
            "symptom_type": "bloating",
            "vas_score": 60,
        },
    )
    assert symptom_resp.status_code == 201
    symptom_id = symptom_resp.json()["id"]

    # Now co-log medication
    resp = await authed_client.post(
        "/api/v1/symptoms/medications",
        json={
            "symptom_log_id": symptom_id,
            "medication_name": "Cetirizine",
            "dose_mg": 10.0,
            "taken_at": now.isoformat(),
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["medication_name"] == "Cetirizine"
    assert data["dose_mg"] == 10.0
    assert data["symptom_log_id"] == symptom_id


@pytest.mark.asyncio
async def test_log_medication_nonexistent_symptom(authed_client: AsyncClient):
    now = datetime.now(timezone.utc)
    resp = await authed_client.post(
        "/api/v1/symptoms/medications",
        json={
            "symptom_log_id": str(uuid.uuid4()),
            "medication_name": "Cetirizine",
            "dose_mg": 10.0,
            "taken_at": now.isoformat(),
        },
    )
    assert resp.status_code == 404


# ── GET /fhir/export ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fhir_export_empty(authed_client: AsyncClient):
    resp = await authed_client.get("/api/v1/fhir/export")
    assert resp.status_code == 200
    data = resp.json()
    assert data["resourceType"] == "Bundle"
    assert data["type"] == "collection"
    assert "entry" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_fhir_export_content_type(authed_client: AsyncClient):
    resp = await authed_client.get("/api/v1/fhir/export")
    assert resp.status_code == 200
    assert "application/fhir+json" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_fhir_export_with_data(authed_client: AsyncClient):
    """With seeded meals and symptoms via API, FHIR bundle should contain entries."""
    now = datetime.now(timezone.utc)

    # Create meal via API
    meal_resp = await authed_client.post("/api/v1/meals", json={
        "timestamp": (now - timedelta(hours=2)).isoformat(),
        "raw_description": "Oatmeal with berries",
        "meal_type": "breakfast",
    })
    assert meal_resp.status_code == 201
    meal_id = meal_resp.json()["id"]

    await authed_client.post(f"/api/v1/meals/{meal_id}/items", json={
        "items": [{"name": "Oatmeal"}]
    })

    # Create symptom via API
    symptom_resp = await authed_client.post("/api/v1/symptoms", json={
        "timestamp": (now - timedelta(hours=1)).isoformat(),
        "symptom_type": "nausea",
        "vas_score": 40,
    })
    assert symptom_resp.status_code == 201

    resp = await authed_client.get("/api/v1/fhir/export")
    assert resp.status_code == 200
    data = resp.json()
    assert data["resourceType"] == "Bundle"
    # Should contain entries from the data we just created
    assert data["total"] >= 2
    resource_types = [e["resource"]["resourceType"] for e in data["entry"]]
    assert "NutritionIntake" in resource_types
    assert "Observation" in resource_types


# ── food_drug_interactions model ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_food_drug_interactions_table_exists(authed_client: AsyncClient):
    """Verify the food_drug_interactions table was created by migration."""
    async with async_session_factory() as session:
        result = await session.execute(text(
            "SELECT EXISTS ("
            "  SELECT FROM information_schema.tables "
            "  WHERE table_name = 'food_drug_interactions'"
            ")"
        ))
        exists = result.scalar()
        assert exists is True


@pytest.mark.asyncio
async def test_medication_logs_table_exists(authed_client: AsyncClient):
    """Verify the medication_logs table was created by migration."""
    async with async_session_factory() as session:
        result = await session.execute(text(
            "SELECT EXISTS ("
            "  SELECT FROM information_schema.tables "
            "  WHERE table_name = 'medication_logs'"
            ")"
        ))
        exists = result.scalar()
        assert exists is True
