"""Tests for the doctor/patient summary signal seam + share endpoint (W2, Pillar 4).

Three layers:
  * Pure derivation unit tests (no DB): direction / confidence / demotion / OR-CI from
    the classical 2x2, straight off ``GuardrailResult``.
  * A contract-guard test asserting the report/export layer references NONE of the
    current engine's output shape or a hard-coded lag window.
  * Async endpoint smoke tests for ``/summary-pdf`` and the signed-share flow.
"""

import inspect
import re

import pytest
from httpx import AsyncClient

from app.models.enums import ComponentType
from app.services import summary_report_service
from app.services.assoc_guardrail import GuardrailResult
from app.services.summary_signal import (
    SummarySignalRow,
    derive_signal_row,
    odds_ratio_ci,
)


def _guard(a, b, c, d, *, test="fisher", p_value=0.5, significant=False):
    return GuardrailResult(
        component_type=ComponentType.HISTAMINES,
        a=a, b=b, c=c, d=d,
        test=test,
        p_value=p_value,
        odds_ratio=None,
        chi2_stat=None,
        min_expected=None,
        q_value=p_value,
        significant=significant,
    )


# ── odds_ratio_ci ──────────────────────────────────────────────────────────────

def test_odds_ratio_ci_matches_hand_computation():
    # [[20, 10],[10, 20]] -> OR = 4.0; log-OR SE = sqrt(1/20+1/10+1/10+1/20)=sqrt(0.3)
    or_, lo, hi = odds_ratio_ci(20, 10, 10, 20)
    assert abs(or_ - 4.0) < 1e-9
    assert lo < 4.0 < hi
    assert lo > 1.0  # a clear elevation excludes 1.0


def test_odds_ratio_ci_haldane_on_zero_cell_stays_finite():
    or_, lo, hi = odds_ratio_ci(18, 23, 0, 1)  # the Garlic-style zero-control cell
    for v in (or_, lo, hi):
        assert v == v and v not in (float("inf"), float("-inf"))  # finite, non-NaN


# ── direction ──────────────────────────────────────────────────────────────────

def test_direction_trigger_when_ci_above_one():
    row = derive_signal_row("Cheddar", _guard(20, 10, 10, 20, significant=True, p_value=0.001))
    assert row.direction == "trigger"


def test_direction_protective_when_ci_below_one():
    # OR = (5*20)/(25*20)?? build a clean protective table: [[5,25],[20,10]] OR=0.1
    row = derive_signal_row("Rice", _guard(5, 25, 20, 10, significant=True, p_value=0.001))
    assert row.direction == "protective"


def test_direction_inconclusive_when_ci_straddles_one():
    row = derive_signal_row("Toast", _guard(6, 6, 5, 6))
    assert row.direction == "inconclusive"


def test_skipped_test_is_inconclusive_insufficient_and_demoted():
    row = derive_signal_row("Water", _guard(0, 10, 0, 5, test="skipped", p_value=None))
    assert row.direction == "inconclusive"
    assert row.confidence == "insufficient"
    assert row.demoted is True
    assert row.odds_ratio is None and row.ci_low is None and row.ci_high is None
    assert row.test == "skipped"


# ── confidence tiers ────────────────────────────────────────────────────────────

def test_strong_requires_significance_and_adequate_counts():
    row = derive_signal_row("Cheddar", _guard(20, 10, 10, 20, significant=True, p_value=0.001))
    assert row.confidence == "strong"
    assert row.demoted is False  # a strong trigger is a confirmed headline


def test_significant_but_thin_is_moderate():
    # significant but only 2 exposed days -> not adequate -> moderate
    row = derive_signal_row("Kimchi", _guard(2, 0, 0, 8, test="fisher",
                                             significant=True, p_value=0.02))
    assert row.confidence == "moderate"


def test_testable_not_significant_is_preliminary_and_demoted():
    row = derive_signal_row("Banana", _guard(6, 6, 5, 7, significant=False, p_value=0.3))
    assert row.confidence == "preliminary"
    assert row.demoted is True


# ── demotion honesty ────────────────────────────────────────────────────────────

def test_protective_food_is_always_demoted():
    row = derive_signal_row("Rice", _guard(5, 25, 20, 10, significant=True, p_value=0.001))
    assert row.demoted is True
    assert row.demotion_reason and "protective" in row.demotion_reason.lower()


def test_sample_counts_come_from_the_2x2_margins():
    row = derive_signal_row("Cheddar", _guard(7, 3, 2, 18))
    assert row.exposed_count == 10   # a + b
    assert row.control_count == 20   # c + d
    assert row.symptom_episodes == 9  # a + c


# ── contract guard: no engine-shape / lag leakage in the report layer ───────────

def test_report_layer_has_no_engine_shape_or_lag_leakage():
    src = inspect.getsource(summary_report_service)
    banned = [
        "combined_score",
        "_driver_for_food",
        "P(beta>0)",
        "trigger_probability",
        "get_suspect_foods",
        "hierarchical",
    ]
    for token in banned:
        assert token not in src, f"report layer must not reference {token!r}"
    # No hard-coded lag window (24/36/48/72h) in the report/export layer.
    assert not re.search(r"\b(24|36|48|72)\s*h", src, re.IGNORECASE)


# ── async endpoint smoke ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summary_pdf_endpoint_returns_pdf(authed_client: AsyncClient):
    resp = await authed_client.get("/api/v1/reports/summary-pdf?lookback_days=30")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:4] == b"%PDF"


@pytest.mark.asyncio
async def test_summary_share_flow_signed_url(authed_client: AsyncClient):
    # 1. mint a signed share URL
    resp = await authed_client.post("/api/v1/reports/summary/share?lookback_days=14")
    assert resp.status_code == 200
    body = resp.json()
    assert "share_url" in body and "token=" in body["share_url"]
    assert body["expires_in_seconds"] == 900

    # 2. the token is the credential — fetch WITHOUT an auth header
    path = body["share_url"].split("http://test", 1)[-1]
    async with AsyncClient(
        transport=authed_client._transport, base_url="http://test"
    ) as anon:
        dl = await anon.get(path)
    assert dl.status_code == 200
    assert dl.headers["content-type"] == "application/pdf"
    assert dl.content[:4] == b"%PDF"


@pytest.mark.asyncio
async def test_summary_share_rejects_tampered_token(authed_client: AsyncClient):
    resp = await authed_client.post("/api/v1/reports/summary/share")
    token = resp.json()["share_url"].split("token=", 1)[-1]
    tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    dl = await authed_client.get(f"/api/v1/reports/summary/shared?token={tampered}")
    assert dl.status_code in (400, 403)
