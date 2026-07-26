"""Wave 2, Pillar 2 — capture-side precise-timestamp tests.

Covers the additive/non-breaking capture layer only:
  * future-timestamp rejection (meals + symptoms)
  * timezone-aware requirement (naive datetimes rejected)
  * occurred-at vs logged-at (timestamp vs created_at) are distinct
  * symptom onset_at capture + onset<=timestamp guard
  * occurred-at defaults to "now" when omitted
  * additive capture fields (client_timezone, time_precision) round-trip
  * backward-compat: create calls that omit all new fields still succeed
"""

from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient


def _future_iso(minutes: int = 120) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


def _past_iso(minutes: int = 120) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


# ---------------------------------------------------------------------------
# Meals
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_meal_backward_compat_no_new_fields(authed_client: AsyncClient):
    """Existing-shape payload (no new capture fields) still succeeds unchanged."""
    resp = await authed_client.post(
        "/api/v1/meals",
        json={
            "timestamp": "2026-04-04T12:30:00Z",
            "meal_type": "lunch",
            "raw_description": "Grilled chicken salad",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    # New fields are additive: client_timezone null when omitted; time_precision
    # falls back to the "exact" server default.
    assert data["client_timezone"] is None
    assert data["time_precision"] == "exact"


@pytest.mark.asyncio
async def test_meal_omitted_timestamp_defaults_to_now(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/meals",
        json={"meal_type": "snack", "raw_description": "handful of almonds"},
    )
    assert resp.status_code == 201
    ts = datetime.fromisoformat(resp.json()["timestamp"])
    assert abs((datetime.now(timezone.utc) - ts).total_seconds()) < 120


@pytest.mark.asyncio
async def test_meal_rejects_future_timestamp(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/meals",
        json={"timestamp": _future_iso(), "meal_type": "lunch"},
    )
    assert resp.status_code == 422
    assert "future" in resp.text.lower()


@pytest.mark.asyncio
async def test_meal_rejects_naive_timestamp(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/meals",
        json={"timestamp": "2026-04-04T12:30:00", "meal_type": "lunch"},
    )
    assert resp.status_code == 422
    assert "timezone" in resp.text.lower()


@pytest.mark.asyncio
async def test_meal_captures_precision_fields(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/meals",
        json={
            "timestamp": _past_iso(),
            "meal_type": "dinner",
            "raw_description": "leftovers",
            "client_timezone": "America/New_York",
            "time_precision": "approximate",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["client_timezone"] == "America/New_York"
    assert data["time_precision"] == "approximate"


@pytest.mark.asyncio
async def test_meal_rejects_bad_precision_value(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/meals",
        json={"timestamp": _past_iso(), "meal_type": "lunch", "time_precision": "guess"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_meal_occurred_at_distinct_from_logged_at(authed_client: AsyncClient):
    """timestamp (occurred-at) must be preserved as sent, not overwritten by insert time."""
    occurred = _past_iso(minutes=600)
    resp = await authed_client.post(
        "/api/v1/meals",
        json={"timestamp": occurred, "meal_type": "breakfast"},
    )
    assert resp.status_code == 201
    data = resp.json()
    occurred_dt = datetime.fromisoformat(occurred)
    stored_dt = datetime.fromisoformat(data["timestamp"])
    assert stored_dt == occurred_dt
    # created_at (logged-at) is server "now", well after the 10h-ago occurred time.
    logged_dt = datetime.fromisoformat(data["created_at"])
    assert logged_dt > stored_dt


# ---------------------------------------------------------------------------
# Symptoms
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_symptom_backward_compat_no_new_fields(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/symptoms",
        json={"timestamp": "2026-04-04T12:30:00Z", "symptom_type": "nausea", "vas_score": 5},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["onset_at"] is None
    assert data["client_timezone"] is None
    # time_precision falls back to the "exact" server default when omitted.
    assert data["time_precision"] == "exact"


@pytest.mark.asyncio
async def test_symptom_omitted_timestamp_defaults_to_now(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/symptoms",
        json={"symptom_type": "bloating", "vas_score": 3},
    )
    assert resp.status_code == 201
    ts = datetime.fromisoformat(resp.json()["timestamp"])
    assert abs((datetime.now(timezone.utc) - ts).total_seconds()) < 120


@pytest.mark.asyncio
async def test_symptom_rejects_future_timestamp(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/symptoms",
        json={"timestamp": _future_iso(), "symptom_type": "pain", "vas_score": 4},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_symptom_rejects_naive_timestamp(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/symptoms",
        json={"timestamp": "2026-04-04T12:30:00", "symptom_type": "pain", "vas_score": 4},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_symptom_captures_onset_at(authed_client: AsyncClient):
    onset = _past_iso(minutes=180)
    observed = _past_iso(minutes=60)
    resp = await authed_client.post(
        "/api/v1/symptoms",
        json={
            "timestamp": observed,
            "onset_at": onset,
            "symptom_type": "headache",
            "vas_score": 6,
            "client_timezone": "Europe/London",
            "time_precision": "exact",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert datetime.fromisoformat(data["onset_at"]) == datetime.fromisoformat(onset)
    assert data["client_timezone"] == "Europe/London"
    assert data["time_precision"] == "exact"


@pytest.mark.asyncio
async def test_symptom_rejects_onset_after_timestamp(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/symptoms",
        json={
            "timestamp": _past_iso(minutes=180),
            "onset_at": _past_iso(minutes=60),  # onset later than observation
            "symptom_type": "fatigue",
            "vas_score": 2,
        },
    )
    assert resp.status_code == 422
    assert "onset" in resp.text.lower()


@pytest.mark.asyncio
async def test_symptom_rejects_future_onset(authed_client: AsyncClient):
    resp = await authed_client.post(
        "/api/v1/symptoms",
        json={
            "timestamp": _past_iso(),
            "onset_at": _future_iso(),
            "symptom_type": "pain",
            "vas_score": 4,
        },
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# UTC normalization (QA gap-closure — non-UTC offset inputs must be stored UTC)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_meal_normalizes_non_utc_offset_to_utc(authed_client: AsyncClient):
    """A non-UTC offset must be stored as the equivalent UTC instant."""
    # 08:30 at +05:00 == 03:30Z the same day.
    resp = await authed_client.post(
        "/api/v1/meals",
        json={
            "timestamp": "2026-04-04T08:30:00+05:00",
            "meal_type": "breakfast",
        },
    )
    assert resp.status_code == 201
    stored = datetime.fromisoformat(resp.json()["timestamp"])
    expected = datetime(2026, 4, 4, 3, 30, tzinfo=timezone.utc)
    assert stored == expected  # same instant
    assert stored.utcoffset() == timedelta(0)  # persisted in UTC, not the source offset


@pytest.mark.asyncio
async def test_symptom_normalizes_non_utc_offset_to_utc(authed_client: AsyncClient):
    """Both timestamp and onset_at supplied in a non-UTC offset store as UTC."""
    resp = await authed_client.post(
        "/api/v1/symptoms",
        json={
            "timestamp": "2026-04-04T08:30:00-04:00",   # == 12:30Z
            "onset_at": "2026-04-04T07:30:00-04:00",     # == 11:30Z (before timestamp)
            "symptom_type": "nausea",
            "vas_score": 5,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    ts = datetime.fromisoformat(data["timestamp"])
    onset = datetime.fromisoformat(data["onset_at"])
    assert ts == datetime(2026, 4, 4, 12, 30, tzinfo=timezone.utc)
    assert onset == datetime(2026, 4, 4, 11, 30, tzinfo=timezone.utc)
    assert ts.utcoffset() == timedelta(0)
    assert onset.utcoffset() == timedelta(0)
