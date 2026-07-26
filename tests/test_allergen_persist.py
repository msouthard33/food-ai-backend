"""Tests for allergen/sensitivity persistence endpoint.

Covers BLK-MOBILE-ALLERGEN-PERSIST: PUT/GET /api/v1/users/me/allergens.

Scenarios:
  - happy path (persist a multi-select, returns updated profile)
  - auth required (no JWT -> 401/422)
  - idempotent re-submit (same set twice = same state, no dup rows)
  - full-replace semantics (changing the set removes de-selected allergens)
  - invalid allergen rejected (422)
  - object form with severity/confirmed metadata
  - GET reflects persisted set
"""

import pytest


ENDPOINT = "/api/v1/users/me/allergens"


async def test_set_allergens_happy_path(authed_client):
    resp = await authed_client.put(
        ENDPOINT, json={"allergens": ["peanuts", "shellfish", "milk_dairy"]}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 3
    returned = {a["allergen"] for a in body["allergens"]}
    assert returned == {"peanuts", "shellfish", "milk_dairy"}


async def test_set_allergens_requires_auth(client):
    resp = await client.put(ENDPOINT, json={"allergens": ["peanuts"]})
    assert resp.status_code in (401, 422)


async def test_get_allergens_requires_auth(client):
    resp = await client.get(ENDPOINT)
    assert resp.status_code in (401, 422)


async def test_set_allergens_idempotent(authed_client):
    payload = {"allergens": ["gluten", "tree_nuts"]}
    first = await authed_client.put(ENDPOINT, json=payload)
    assert first.status_code == 200, first.text
    second = await authed_client.put(ENDPOINT, json=payload)
    assert second.status_code == 200, second.text
    # Same set, no duplicate rows.
    assert second.json()["count"] == 2
    assert {a["allergen"] for a in second.json()["allergens"]} == {"gluten", "tree_nuts"}


async def test_set_allergens_full_replace(authed_client):
    await authed_client.put(ENDPOINT, json={"allergens": ["soy", "eggs", "fish"]})
    resp = await authed_client.put(ENDPOINT, json={"allergens": ["soy"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
    assert body["allergens"][0]["allergen"] == "soy"


async def test_set_allergens_empty_clears(authed_client):
    await authed_client.put(ENDPOINT, json={"allergens": ["sesame"]})
    resp = await authed_client.put(ENDPOINT, json={"allergens": []})
    assert resp.status_code == 200, resp.text
    assert resp.json()["count"] == 0


async def test_set_allergens_invalid_rejected(authed_client):
    resp = await authed_client.put(
        ENDPOINT, json={"allergens": ["peanuts", "not_a_real_allergen"]}
    )
    assert resp.status_code == 422, resp.text


async def test_set_allergens_duplicate_rejected(authed_client):
    resp = await authed_client.put(
        ENDPOINT, json={"allergens": ["peanuts", "peanuts"]}
    )
    assert resp.status_code == 422, resp.text


async def test_set_allergens_object_form_metadata(authed_client):
    resp = await authed_client.put(
        ENDPOINT,
        json={
            "allergens": [
                {"allergen": "peanuts", "confirmed": True, "severity": "severe"},
                "shellfish",
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = {a["allergen"]: a for a in resp.json()["allergens"]}
    assert body["peanuts"]["confirmed"] is True
    assert body["peanuts"]["severity"] == "severe"
    assert body["shellfish"]["confirmed"] is False
    assert body["shellfish"]["severity"] is None


async def test_set_allergens_invalid_severity_rejected(authed_client):
    resp = await authed_client.put(
        ENDPOINT,
        json={"allergens": [{"allergen": "peanuts", "severity": "catastrophic"}]},
    )
    assert resp.status_code == 422, resp.text


async def test_get_allergens_reflects_persisted_set(authed_client):
    await authed_client.put(ENDPOINT, json={"allergens": ["histamines", "sulfites"]})
    resp = await authed_client.get(ENDPOINT)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {a["allergen"] for a in body["allergens"]} == {"histamines", "sulfites"}
    assert body["count"] == 2
