"""OQ-IOS-SHARE-SCHEME: signed share/download URLs must reflect the public
https scheme when the app runs behind a TLS-terminating proxy (Railway), which
forwards plain http internally with ``X-Forwarded-Proto: https``.

These tests exercise the ASGI ``ProxyHeadersMiddleware`` wired in ``app.main``:
  * with ``X-Forwarded-Proto: https`` -> builder must emit ``https://``
  * without any forwarded header (local dev) -> plain ``http://`` still works

Every endpoint that returns a ``request.url_for(...)``-derived URL is covered so
the fix is verified uniformly, not one-off.
"""

import uuid

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


def _client_for(uid: uuid.UUID, forwarded_proto: str | None = None) -> AsyncClient:
    headers = {"Authorization": f"Bearer {uid}"}
    if forwarded_proto is not None:
        headers["X-Forwarded-Proto"] = forwarded_proto
    # base_url uses http:// to emulate the internal proxy->app hop.
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers=headers,
    )


# ── summary-report share URL ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summary_share_url_https_behind_proxy():
    uid = await _new_user()
    async with _client_for(uid, forwarded_proto="https") as client:
        resp = await client.post("/api/v1/reports/summary/share")
        assert resp.status_code == 200
        share_url = resp.json()["share_url"]
        assert share_url.startswith("https://"), share_url
        assert "/api/v1/reports/summary/shared" in share_url
        assert "token=" in share_url


@pytest.mark.asyncio
async def test_summary_share_url_http_local_dev():
    """No forwarded header (local dev) -> http scheme preserved, URL still works."""
    uid = await _new_user()
    async with _client_for(uid) as client:
        resp = await client.post("/api/v1/reports/summary/share")
        assert resp.status_code == 200
        share_url = resp.json()["share_url"]
        assert share_url.startswith("http://"), share_url
        assert "/api/v1/reports/summary/shared" in share_url


# ── data-export download URL ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_export_download_url_https_behind_proxy():
    uid = await _new_user()
    async with _client_for(uid, forwarded_proto="https") as client:
        resp = await client.post("/api/v1/export/request")
        assert resp.status_code == 200
        download_url = resp.json()["download_url"]
        assert download_url.startswith("https://"), download_url
        assert "/api/v1/export/download" in download_url
        assert "token=" in download_url


@pytest.mark.asyncio
async def test_export_download_url_http_local_dev():
    uid = await _new_user()
    async with _client_for(uid) as client:
        resp = await client.post("/api/v1/export/request")
        assert resp.status_code == 200
        download_url = resp.json()["download_url"]
        assert download_url.startswith("http://"), download_url
        assert "/api/v1/export/download" in download_url
