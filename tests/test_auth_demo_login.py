"""Tests for the demo-login auth endpoint.

The demo-login endpoint lets the mobile app enter a fully-authenticated demo
session without a working Supabase project. These tests cover:

* a token is issued and is accepted by a real authenticated endpoint,
* the demo identity is deterministic across calls,
* when a JWT secret is configured, the minted HS256 claims match exactly what
  ``get_current_user`` decodes, and
* the endpoint is disabled when the feature flag is off.
"""

from jose import jwt

from app.config import get_settings
from app.routers.auth import demo_user_email, demo_user_id


async def test_demo_login_issues_token_accepted_by_authed_endpoint(client):
    """A demo token is returned and works against an authenticated endpoint."""
    resp = await client.post("/api/v1/auth/demo-login", json={"persona": "ibs"})
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["token_type"] == "bearer"
    assert data["persona"] == "ibs"
    assert data["access_token"]
    assert data["expires_in"] > 0
    assert data["user_id"] == str(demo_user_id("ibs"))

    # The token must be accepted by a genuinely authenticated endpoint.
    authed = await client.get(
        "/api/v1/insights/triggers",
        headers={"Authorization": f"Bearer {data['access_token']}"},
    )
    assert authed.status_code == 200, authed.text


async def test_demo_login_seeds_day_one_priors(client):
    """A fresh demo login already has cold-start trigger priors."""
    resp = await client.post("/api/v1/auth/demo-login", json={"persona": "mcas"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]

    triggers = await client.get(
        "/api/v1/insights/triggers",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert triggers.status_code == 200, triggers.text
    # MCAS maps to 3 component priors (histamines, salicylates, oxalates).
    assert triggers.json()["total"] >= 1


async def test_demo_login_is_deterministic_across_calls(client):
    """Same persona -> same user_id across repeated calls."""
    first = await client.post("/api/v1/auth/demo-login", json={"persona": "ibs"})
    second = await client.post("/api/v1/auth/demo-login", json={"persona": "ibs"})
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["user_id"] == second.json()["user_id"]
    assert first.json()["user_id"] == str(demo_user_id("ibs"))


async def test_demo_login_defaults_to_ibs(client):
    """Empty body defaults persona to ibs."""
    resp = await client.post("/api/v1/auth/demo-login", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["persona"] == "ibs"


async def test_minted_claims_match_decoder(client, monkeypatch):
    """With a JWT secret set, minted HS256 claims match get_current_user's decode.

    get_current_user (_decode_supabase_jwt) decodes HS256 with the secret and
    ``audience="authenticated"`` and reads ``sub`` (+ ``email`` for provisioning).
    """
    secret = "test-demo-login-secret"
    settings = get_settings()  # lru_cached singleton shared with app.dependencies
    monkeypatch.setattr(settings, "supabase_jwt_secret", secret)

    resp = await client.post("/api/v1/auth/demo-login", json={"persona": "ibs"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]

    # It is a real JWT now (not the raw-UUID dev token).
    assert token.count(".") == 2

    # Decode exactly the way the dependency does.
    claims = jwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        audience="authenticated",
    )
    assert claims["sub"] == str(demo_user_id("ibs"))
    assert claims["aud"] == "authenticated"
    assert claims["email"] == demo_user_email("ibs")
    assert "exp" in claims and "iat" in claims

    # And the token is accepted end-to-end by an authenticated endpoint.
    authed = await client.get(
        "/api/v1/insights/triggers",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert authed.status_code == 200, authed.text


async def test_demo_login_disabled_when_flag_off(client, monkeypatch):
    """Flag off -> endpoint returns 404."""
    settings = get_settings()
    monkeypatch.setattr(settings, "demo_login_enabled", False)

    resp = await client.post("/api/v1/auth/demo-login", json={"persona": "ibs"})
    assert resp.status_code == 404, resp.text
