"""W2-C tests: clinician PDF report (Box 11) + mySymptoms CSV import (Box 12).

Integration tests use fresh per-test users so aggregate assertions are isolated from
data other tests leave for the shared TEST_USER_ID.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.main import app
from tests.conftest import _ensure_tables, async_session_factory


async def _new_user() -> uuid.UUID:
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


async def _seed_trigger(client: AsyncClient, n_symptoms: int, food: str) -> list[str]:
    now = datetime.now(timezone.utc)
    meal_resp = await client.post(
        "/api/v1/meals",
        json={
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "raw_description": f"{food} plate",
            "meal_type": "lunch",
        },
    )
    assert meal_resp.status_code == 201
    meal_id = meal_resp.json()["id"]
    await client.post(f"/api/v1/meals/{meal_id}/items", json={"items": [{"name": food}]})

    symptom_ids: list[str] = []
    for i in range(n_symptoms):
        resp = await client.post(
            "/api/v1/symptoms",
            json={
                "timestamp": (now - timedelta(minutes=30 * i)).isoformat(),
                "symptom_type": "bloating",
                "vas_score": 70,
            },
        )
        assert resp.status_code == 201
        symptom_ids.append(resp.json()["id"])
    return symptom_ids


# ── Box 11: clinician PDF ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clinician_pdf_generates_with_data():
    uid = await _new_user()
    async with _client_for(uid) as client:
        symptom_ids = await _seed_trigger(client, n_symptoms=3, food="Zzz PDF Trigger")
        # medication co-log so the confound column exercises both branches
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

        resp = await client.get("/api/v1/reports/clinician-pdf")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert "attachment" in resp.headers.get("content-disposition", "")
        assert resp.content[:5] == b"%PDF-"
        # A non-trivial multi-section PDF is well over a few hundred bytes.
        assert len(resp.content) > 1500


@pytest.mark.asyncio
async def test_clinician_pdf_generates_with_no_data():
    """Day-one users have no history; the PDF must still render (no empty-state error)."""
    uid = await _new_user()
    async with _client_for(uid) as client:
        resp = await client.get("/api/v1/reports/clinician-pdf?lookback_days=14")
        assert resp.status_code == 200
        assert resp.content[:5] == b"%PDF-"


@pytest.mark.asyncio
async def test_clinician_pdf_requires_auth():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer not-a-valid-token"},
    ) as client:
        resp = await client.get("/api/v1/reports/clinician-pdf")
        assert resp.status_code == 401


# ── Box 12: mySymptoms CSV import ────────────────────────────────────────────

_MYSYMPTOMS_CSV = (
    "Date,Time,Type,Name,Severity,Note\n"
    "2026-07-01,08:30,Food,Scrambled eggs; toast,,Breakfast at home\n"
    "2026-07-01,12:15,Drink,Coffee,,\n"
    "2026-07-01,14:00,Symptom,Bloating,6,Felt uncomfortable\n"
    "2026-07-01,15:30,Symptom,Headache,4,\n"
    "2026-07-01,18:00,Food,Grilled chicken salad,,\n"
    "2026-07-01,20:00,Medication,Loratadine,,10mg\n"
)


def _files(csv_text: str):
    return {"file": ("mysymptoms_export.csv", csv_text.encode("utf-8"), "text/csv")}


@pytest.mark.asyncio
async def test_csv_import_creates_meals_and_symptoms():
    uid = await _new_user()
    async with _client_for(uid) as client:
        resp = await client.post("/api/v1/import/csv", files=_files(_MYSYMPTOMS_CSV))
        assert resp.status_code == 200
        data = resp.json()

        assert data["source_format"] == "mysymptoms"
        assert data["total_rows"] == 6
        assert data["meals_created"] == 3  # 2 foods + 1 drink
        assert data["symptoms_created"] == 2
        # The Medication row is unsupported -> skipped with a reason.
        assert data["rows_skipped"] == 1
        assert any("Medication" in e["reason"] for e in data["errors"])

        # Verify the data actually landed via the meals list endpoint.
        meals = (await client.get("/api/v1/meals")).json()
        names = {
            item["name"]
            for meal in meals["items"]
            for item in meal["items"]
        }
        assert "Scrambled eggs" in names
        assert "toast" in names
        assert "Coffee" in names


@pytest.mark.asyncio
async def test_csv_import_is_idempotent():
    uid = await _new_user()
    async with _client_for(uid) as client:
        first = (await client.post("/api/v1/import/csv", files=_files(_MYSYMPTOMS_CSV))).json()
        assert first["meals_created"] == 3
        assert first["symptoms_created"] == 2

        second = (await client.post("/api/v1/import/csv", files=_files(_MYSYMPTOMS_CSV))).json()
        # Re-import creates nothing new; every diary row is recognized as a duplicate.
        assert second["meals_created"] == 0
        assert second["symptoms_created"] == 0
        assert second["rows_skipped"] == 6


@pytest.mark.asyncio
async def test_csv_import_reports_row_errors():
    uid = await _new_user()
    bad_csv = (
        "Date,Time,Type,Name,Severity,Note\n"
        "not-a-date,08:30,Food,Eggs,,\n"
        "2026-07-02,09:00,Food,,,missing name\n"
        "2026-07-02,10:00,Symptom,Nausea,3,\n"
    )
    async with _client_for(uid) as client:
        data = (await client.post("/api/v1/import/csv", files=_files(bad_csv))).json()
        assert data["total_rows"] == 3
        assert data["symptoms_created"] == 1
        assert data["meals_created"] == 0
        assert data["rows_skipped"] == 2
        reasons = " ".join(e["reason"] for e in data["errors"])
        assert "date" in reasons.lower()
        assert "name" in reasons.lower()


@pytest.mark.asyncio
async def test_csv_import_severity_and_symptom_mapping():
    uid = await _new_user()
    csv_text = (
        "Date,Time,Type,Name,Severity,Note\n"
        "2026-07-03,11:00,Symptom,Diarrhea,8,\n"
        "2026-07-03,12:00,Symptom,Brain fog,5,\n"
    )
    async with _client_for(uid) as client:
        data = (await client.post("/api/v1/import/csv", files=_files(csv_text))).json()
        assert data["symptoms_created"] == 2

        symptoms = (await client.get("/api/v1/symptoms")).json()
        by_type = {s["symptom_type"]: s for s in symptoms["items"]}
        # 0-10 severity scaled to 0-100 VAS.
        assert by_type["bowel_changes"]["vas_score"] == 80
        assert by_type["brain_fog"]["vas_score"] == 50


@pytest.mark.asyncio
async def test_csv_import_empty_file_rejected():
    uid = await _new_user()
    async with _client_for(uid) as client:
        resp = await client.post("/api/v1/import/csv", files=_files(""))
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_csv_import_missing_columns_reported():
    uid = await _new_user()
    csv_text = "Foo,Bar\n1,2\n"
    async with _client_for(uid) as client:
        data = (await client.post("/api/v1/import/csv", files=_files(csv_text))).json()
        assert data["meals_created"] == 0
        assert data["symptoms_created"] == 0
        assert any("Missing required column" in e["reason"] for e in data["errors"])
