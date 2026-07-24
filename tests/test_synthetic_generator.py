"""Tests for synthetic_data_generator.py

All tests are pure Python — no database connection required.
Tests verify clinical correctness of the lag distributions, food name fidelity,
diary structure, and the decay threshold constant.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

# Resolve KB path relative to this test file (backend/tests/ → project root)
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.normpath(
    os.path.join(_TESTS_DIR, "../../04 - Food Science & Data/allergen_knowledge_base_complete.json")
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def kb_data():
    from app.services.synthetic_data_generator import load_kb_food_index
    kb_index, safe_foods = load_kb_food_index(KB_PATH)
    assert kb_index, "KB index should not be empty"
    assert safe_foods, "Safe foods list should not be empty"
    return kb_index, safe_foods


@pytest.fixture(scope="module")
def ibs_profile(kb_data):
    from app.services.synthetic_data_generator import generate_patient_profile
    kb_index, safe_foods = kb_data
    return generate_patient_profile(
        conditions=["ibs"],
        patient_index=0,
        seed=42,
        kb_food_index=kb_index,
        safe_foods=safe_foods,
    )


@pytest.fixture(scope="module")
def ibs_diary(ibs_profile, kb_data):
    from app.services.synthetic_data_generator import generate_patient_diary
    kb_index, _ = kb_data
    start_date = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    return generate_patient_diary(ibs_profile, kb_index, start_date, num_weeks=8)


@pytest.fixture(scope="module")
def mcas_profile(kb_data):
    from app.services.synthetic_data_generator import generate_patient_profile
    kb_index, safe_foods = kb_data
    return generate_patient_profile(
        conditions=["mcas"],
        patient_index=10,
        seed=42,
        kb_food_index=kb_index,
        safe_foods=safe_foods,
    )


@pytest.fixture(scope="module")
def mcas_diary(mcas_profile, kb_data):
    from app.services.synthetic_data_generator import generate_patient_diary
    kb_index, _ = kb_data
    # Set logging_consistency to 1.0 for a full sample
    profile_full_log = {**mcas_profile, "logging_consistency": 1.0}
    start_date = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    return generate_patient_diary(profile_full_log, kb_index, start_date, num_weeks=8)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_ibs_profile_has_fodmap_trigger(ibs_profile):
    """IBS profile must include at least one FODMAP-family trigger component."""
    fodmap_family = {"fodmap", "lactose", "fructose", "sorbitol"}
    trigger_components = set(ibs_profile["trigger_components"])
    assert trigger_components & fodmap_family, (
        f"IBS profile trigger_components {trigger_components} should overlap "
        f"with FODMAP family {fodmap_family}"
    )


def test_diary_meal_count_8_weeks(ibs_diary):
    """8-week diary should produce 100–250 logged meals after applying logging_consistency."""
    logged_meals = [m for m in ibs_diary["meals"] if m["logged"]]
    count = len(logged_meals)
    assert 100 <= count <= 250, (
        f"Expected 100–250 logged meals for 8 weeks, got {count}"
    )


def test_symptom_timestamps_after_meals(ibs_diary):
    """Every symptom timestamp must be strictly after its triggering meal timestamp."""
    for symptom in ibs_diary["symptoms"]:
        meal_ts = symptom["triggering_meal_timestamp"]
        sym_ts = symptom["timestamp"]
        assert sym_ts > meal_ts, (
            f"Symptom at {sym_ts} is not after triggering meal at {meal_ts}"
        )


def test_lag_within_condition_bounds(ibs_diary):
    """IBS symptom lags must fall within the triangular distribution bounds [2, 36] hours."""
    for symptom in ibs_diary["symptoms"]:
        lag = symptom["lag_hours"]
        assert 2.0 <= lag <= 36.0, (
            f"IBS lag {lag:.3f}h is outside [2.0, 36.0] bounds"
        )


def test_no_fabricated_food_names(kb_data, ibs_profile, ibs_diary):
    """All food names in the diary must come from the KB food index or safe foods list."""
    kb_index, safe_foods = kb_data
    # Build set of all valid KB food names
    all_kb_names: set[str] = set(safe_foods)
    for foods in kb_index.values():
        all_kb_names.update(foods)

    fabricated = []
    for meal in ibs_diary["meals"]:
        for food_name in meal["foods"]:
            if food_name not in all_kb_names:
                fabricated.append(food_name)

    assert not fabricated, (
        f"Found {len(fabricated)} fabricated food names not in KB: {fabricated[:10]}"
    )


def test_mcas_bimodal_lag(mcas_diary):
    """MCAS symptom lags must cluster in the two bimodal ranges [0.08, 0.5] or [2.0, 6.0]."""
    symptoms = mcas_diary["symptoms"]
    if not symptoms:
        pytest.skip("No symptoms generated for MCAS patient — try a different seed")

    lags = [s["lag_hours"] for s in symptoms]
    immediate = sum(1 for lag in lags if 0.0 <= lag <= 0.5)
    delayed = sum(1 for lag in lags if 2.0 <= lag <= 6.0)
    total = len(lags)
    in_bimodal_range = immediate + delayed

    assert in_bimodal_range >= total * 0.95, (
        f"MCAS lags should be almost entirely in bimodal ranges. "
        f"Immediate [0–0.5h]: {immediate}, Delayed [2–6h]: {delayed}, "
        f"Total: {total}, in-range fraction: {in_bimodal_range/total:.2f}"
    )


def test_decay_threshold_constant():
    """SYNTHETIC_DECAY_THRESHOLD must equal 42 (14 days of consistent 3-meal logging)."""
    from app.services.trigger_service import SYNTHETIC_DECAY_THRESHOLD
    assert SYNTHETIC_DECAY_THRESHOLD == 42, (
        f"SYNTHETIC_DECAY_THRESHOLD should be 42, got {SYNTHETIC_DECAY_THRESHOLD}"
    )
