"""W2-B1 tests: insights leaderboard upgrade (Box 8), medication covariate (Box 10),
full data export + soft delete (Box 13).

Integration tests run against fresh, per-test users so aggregate assertions are not
polluted by data other tests leave for the shared TEST_USER_ID.
"""

import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.main import app
from tests.conftest import _ensure_tables, async_session_factory

UNIQUE_FOOD = "Zzz Test Trigger Food"


# ── helpers ──────────────────────────────────────────────────────────────────

async def _new_user() -> uuid.UUID:
    """Insert a fresh user and return its id (isolates aggregate assertions)."""
    await _ensure_tables()
    uid = uuid.uuid4()
    async with async_session_factory() as session:
        await session.execute(
            text(
                "INSERT INTO users (id, email, timezone, onboarding_completed) "
                "VALUES (:id, :email, 'UTC', false)"
            ),
            {"id": uid, "email": f"{uid}@foodai.test"},
        )
        await session.commit()
    return uid


def _client_for(uid: uuid.UUID) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {uid}"},
    )


async def _seed_kb_food(name: str = UNIQUE_FOOD) -> None:
    """Insert a KB FoodEntry (carrying FODMAP at exposure level) named ``name``.

    The Bayesian engine attributes exposure via the food KB — meal items logged
    through the API carry a NULL ``food_entry_id`` and are resolved to KB foods by
    (case-insensitive) name. Idempotent: skips if a FoodEntry with ``name`` exists.
    """
    from decimal import Decimal

    from app.models.enums import ComponentType
    from app.models.food import FoodComponentDetail, FoodEntry

    async with async_session_factory() as session:
        existing = (
            await session.execute(
                text("SELECT id FROM food_database WHERE lower(name) = lower(:n) LIMIT 1"),
                {"n": name},
            )
        ).first()
        if existing is not None:
            return
        food = FoodEntry(name=name)
        session.add(food)
        await session.flush()
        session.add(
            FoodComponentDetail(
                food_entry_id=food.id,
                component_type=ComponentType.FODMAP,
                level=Decimal("3.0"),
            )
        )
        await session.commit()


async def _seed_trigger(client: AsyncClient, n_symptoms: int, food: str = UNIQUE_FOOD) -> list[str]:
    """Seed ``food`` (KB-linked) eaten on ``n_symptoms`` distinct days, each followed
    by a symptom, so the Bayesian engine sees the food as an exposure with outcomes.

    Returns the list of created symptom ids.
    """
    await _seed_kb_food(food)
    now = datetime.now(timezone.utc)

    symptom_ids: list[str] = []
    for i in range(n_symptoms):
        # Distinct calendar day per episode: meal at 09:00, symptom at 14:00 same day.
        day = (now - timedelta(days=i)).replace(hour=9, minute=0, second=0, microsecond=0)
        meal_resp = await client.post(
            "/api/v1/meals",
            json={
                "timestamp": day.isoformat(),
                "raw_description": f"{food} plate",
                "meal_type": "lunch",
            },
        )
        assert meal_resp.status_code == 201
        meal_id = meal_resp.json()["id"]
        await client.post(f"/api/v1/meals/{meal_id}/items", json={"items": [{"name": food}]})

        resp = await client.post(
            "/api/v1/symptoms",
            json={
                "timestamp": day.replace(hour=14).isoformat(),
                "symptom_type": "bloating",
                "vas_score": 70,
            },
        )
        assert resp.status_code == 201
        symptom_ids.append(resp.json()["id"])
    return symptom_ids


# ── unit tests: confidence stats ─────────────────────────────────────────────

def test_wilson_interval_bounds():
    from app.utils.confidence import wilson_interval

    assert wilson_interval(0, 0) == (0.0, 0.0)

    low, high = wilson_interval(1, 1)
    assert 0.0 < low < high
    assert high == 1.0  # capped at 1.0 when all successes

    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high
    assert 0.0 <= low <= high <= 1.0


def test_evidence_confidence_label():
    from app.utils.confidence import evidence_confidence_label

    assert evidence_confidence_label(5, 20.0) == "Strong signal"
    assert evidence_confidence_label(5, 30.0) == "Emerging signal"  # interval too wide
    assert evidence_confidence_label(3, 50.0) == "Emerging signal"
    assert evidence_confidence_label(2, 10.0) == "Preliminary"


def test_medication_adjusted_score():
    from app.services.medication_service import medication_adjusted_score

    assert medication_adjusted_score(100.0, 3, 0) == 100.0  # no meds -> unchanged
    assert medication_adjusted_score(100.0, 3, 3) == 50.0  # all medicated -> -50%
    assert medication_adjusted_score(100.0, 4, 2) == 75.0  # half medicated -> -25%
    assert medication_adjusted_score(100.0, 0, 0) == 100.0  # no episodes -> unchanged


# ── Box 8: suspect-foods leaderboard upgrade ─────────────────────────────────

@pytest.mark.asyncio
async def test_suspect_foods_upgraded_fields():
    uid = await _new_user()
    async with _client_for(uid) as client:
        await _seed_trigger(client, n_symptoms=3)

        resp = await client.get("/api/v1/insights/suspect-foods")
        assert resp.status_code == 200
        foods = resp.json()["foods"]
        row = next(f for f in foods if f["food_name"] == UNIQUE_FOOD)

        # Every required Box-8 field present (now versioned + Bayesian)
        for key in (
            "trigger_score",
            "combined_score",
            "ci_low",
            "ci_high",
            "n_meals",
            "n_symptom_episodes",
            "confidence_label",
            "confidence_tier",
            "method",
            "trigger_probability",
        ):
            assert key in row, f"missing {key}"

        assert row["n_symptom_episodes"] == 3
        assert row["n_meals"] == 3  # one meal per exposed day
        # Versioned Bayesian contract
        assert row["method"] == "bayesian_beta_binomial"
        assert 0.0 <= row["trigger_probability"] <= 1.0
        # Bayesian score (trigger_probability * 100), no longer a raw proportion
        assert 0.0 < row["trigger_score"] <= 100.0
        assert abs(row["trigger_score"] - row["trigger_probability"] * 100.0) < 0.5
        # No medication logged -> combined score equals the raw trigger score
        assert row["combined_score"] == row["trigger_score"]
        assert row["medication_confounded"] is False
        # 95% credible interval is a proper sub-interval of the 0–100 scale
        assert 0.0 <= row["ci_low"] < row["ci_high"] <= 100.0
        # n=3 episodes -> "Emerging signal" (below the >=5-episode "Strong" tier)
        assert row["confidence_label"] == "Emerging signal"


# ── Box 10: medication co-log as correlation covariate ───────────────────────

@pytest.mark.asyncio
async def test_suspect_foods_medication_covariate():
    uid = await _new_user()
    async with _client_for(uid) as client:
        symptom_ids = await _seed_trigger(client, n_symptoms=3)

        # Co-log an antihistamine against exactly one of the symptom episodes
        med_resp = await client.post(
            "/api/v1/symptoms/medications",
            json={
                "symptom_log_id": symptom_ids[0],
                "medication_name": "Cetirizine",
                "dose_mg": 10.0,
                "taken_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        assert med_resp.status_code == 201

        resp = await client.get("/api/v1/insights/suspect-foods")
        assert resp.status_code == 200
        row = next(f for f in resp.json()["foods"] if f["food_name"] == UNIQUE_FOOD)

        assert row["n_medicated_episodes"] == 1
        assert row["medication_confounded"] is True
        # Medication co-log discounts the combined score below the raw trigger score
        assert row["combined_score"] < row["trigger_score"]
        # 1 of 3 medicated -> combined = trigger * (1 - 0.5 * 1/3) ≈ 83.33
        assert abs(row["combined_score"] - row["trigger_score"] * (1 - 0.5 / 3)) < 0.1


@pytest.mark.asyncio
async def test_lag_correlation_medication_fields():
    uid = await _new_user()
    async with _client_for(uid) as client:
        symptom_ids = await _seed_trigger(client, n_symptoms=2)
        await client.post(
            "/api/v1/symptoms/medications",
            json={
                "symptom_log_id": symptom_ids[0],
                "medication_name": "Famotidine",
                "taken_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        resp = await client.get("/api/v1/insights/lag-correlation")
        assert resp.status_code == 200
        rows = [r for r in resp.json()["correlations"] if r["food_name"] == UNIQUE_FOOD]
        assert rows, "expected a lag-correlation row for the seeded food"
        for r in rows:
            assert "n_medicated_episodes" in r
            assert "medication_confounded" in r
        # At least one bucket should reflect the medicated episode
        assert any(r["medication_confounded"] for r in rows)


# ── Box 13: full data export via signed URL ──────────────────────────────────

@pytest.mark.asyncio
async def test_export_request_returns_signed_url():
    uid = await _new_user()
    async with _client_for(uid) as client:
        resp = await client.post("/api/v1/export/request")
        assert resp.status_code == 200
        data = resp.json()
        assert "/api/v1/export/download" in data["download_url"]
        assert "token=" in data["download_url"]
        assert data["expires_in_seconds"] == 900
        assert "expires_at" in data


@pytest.mark.asyncio
async def test_export_download_full_bundle():
    uid = await _new_user()
    async with _client_for(uid) as client:
        symptom_ids = await _seed_trigger(client, n_symptoms=1)
        await client.post(
            "/api/v1/symptoms/medications",
            json={
                "symptom_log_id": symptom_ids[0],
                "medication_name": "Cetirizine",
                "dose_mg": 10.0,
                "taken_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        await client.post("/api/v1/protocols/start", json={"protocol_type": "low-histamine"})

        req = await client.post("/api/v1/export/request")
        download_url = req.json()["download_url"]

        resp = await client.get(download_url)
        assert resp.status_code == 200
        assert "attachment" in resp.headers.get("content-disposition", "")
        data = resp.json()

        assert data["export_version"] == "1.0"
        assert data["user"]["id"] == str(uid)
        assert data["counts"]["meals"] >= 1
        assert data["counts"]["symptoms"] >= 1
        assert data["counts"]["medications"] >= 1
        assert data["counts"]["protocols"] >= 1
        assert any(
            item["name"] == UNIQUE_FOOD
            for meal in data["meals"]
            for item in meal["items"]
        )


@pytest.mark.asyncio
async def test_export_download_invalid_token():
    uid = await _new_user()
    async with _client_for(uid) as client:
        resp = await client.get("/api/v1/export/download?token=abc.def")
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_export_download_expired_token():
    from app.routers.export import _TOKEN_PURPOSE, _sign

    uid = await _new_user()
    expired = _sign({"uid": str(uid), "exp": int(time.time()) - 10, "purpose": _TOKEN_PURPOSE})
    async with _client_for(uid) as client:
        resp = await client.get(f"/api/v1/export/download?token={expired}")
        assert resp.status_code == 410


@pytest.mark.asyncio
async def test_export_soft_delete_excludes_meal():
    uid = await _new_user()
    async with _client_for(uid) as client:
        now = datetime.now(timezone.utc)
        keep = await client.post(
            "/api/v1/meals",
            json={"timestamp": now.isoformat(), "raw_description": "keep", "meal_type": "dinner"},
        )
        drop = await client.post(
            "/api/v1/meals",
            json={"timestamp": now.isoformat(), "raw_description": "drop", "meal_type": "dinner"},
        )
        keep_id = keep.json()["id"]
        drop_id = drop.json()["id"]

        # Soft-delete one meal directly (no soft-delete endpoint in scope for W2-B1)
        async with async_session_factory() as session:
            await session.execute(
                text("UPDATE meals SET deleted_at = now() WHERE id = :id"),
                {"id": uuid.UUID(drop_id)},
            )
            await session.commit()

        req = await client.post("/api/v1/export/request")
        resp = await client.get(req.json()["download_url"])
        assert resp.status_code == 200
        meal_ids = {m["id"] for m in resp.json()["meals"]}
        assert keep_id in meal_ids
        assert drop_id not in meal_ids


@pytest.mark.asyncio
async def test_deleted_at_columns_exist():
    await _ensure_tables()
    async with async_session_factory() as session:
        for table in ("meals", "symptom_scores", "medication_logs", "user_sensitivity_profiles"):
            result = await session.execute(
                text(
                    "SELECT EXISTS (SELECT FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = 'deleted_at')"
                ),
                {"t": table},
            )
            assert result.scalar() is True, f"{table}.deleted_at missing"
