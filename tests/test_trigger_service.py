"""Tests for Sprint C trigger service additions.

All tests are pure Python — no database connection required.
Tests verify:
  - Daily load aggregation (_compute_daily_loads helper)
  - High-load day threshold detection
  - apply_preparation_modifiers delta application and clamping
  - extract_preparation_method keyword splitting
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


# ── Tiny mock objects (no ORM, no DB) ────────────────────────────────────────


@dataclass
class MockComponent:
    component_type: object  # ComponentType enum value
    estimated_level: float | None


@dataclass
class MockItem:
    components: list[MockComponent] = field(default_factory=list)


@dataclass
class MockMeal:
    timestamp: datetime
    items: list[MockItem] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_utc(year: int, month: int, day: int, hour: int = 12) -> datetime:
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


# ── Tests: daily load aggregation ────────────────────────────────────────────


def test_daily_load_aggregation():
    """_compute_daily_loads sums estimated_level for each ComponentType within a calendar day."""
    from app.models.enums import ComponentType
    from app.services.trigger_service import _compute_daily_loads

    # Two meals on the same calendar day, both containing HISTAMINES
    meals = [
        MockMeal(
            timestamp=_make_utc(2026, 7, 1, 8),
            items=[MockItem(components=[MockComponent(ComponentType.HISTAMINES, 35.0)])],
        ),
        MockMeal(
            timestamp=_make_utc(2026, 7, 1, 18),
            items=[MockItem(components=[MockComponent(ComponentType.HISTAMINES, 40.0)])],
        ),
    ]

    daily_loads, _ = _compute_daily_loads(meals, high_load_threshold=60.0)

    day_key = next(iter(daily_loads))
    assert ComponentType.HISTAMINES in daily_loads[day_key], "HISTAMINES should be in daily_loads"
    # 35 + 40 = 75
    assert daily_loads[day_key][ComponentType.HISTAMINES] == 75.0, (
        f"Expected daily sum 75.0, got {daily_loads[day_key][ComponentType.HISTAMINES]}"
    )


def test_high_load_day_detection():
    """Days with daily sum >= threshold are marked high-load; days below threshold are not."""
    from app.models.enums import ComponentType
    from app.services.trigger_service import _compute_daily_loads

    # Day 1: sum = 65 (high load)
    # Day 2: sum = 55 (below threshold)
    meals = [
        MockMeal(
            timestamp=_make_utc(2026, 7, 1),
            items=[MockItem(components=[MockComponent(ComponentType.HISTAMINES, 65.0)])],
        ),
        MockMeal(
            timestamp=_make_utc(2026, 7, 2),
            items=[MockItem(components=[MockComponent(ComponentType.HISTAMINES, 55.0)])],
        ),
    ]

    daily_loads, high_load_days = _compute_daily_loads(meals, high_load_threshold=60.0)

    from datetime import date

    high_days = high_load_days.get(ComponentType.HISTAMINES, [])
    assert date(2026, 7, 1) in high_days, "Day 1 (sum=65) should be a high-load day"
    assert date(2026, 7, 2) not in high_days, "Day 2 (sum=55) should NOT be a high-load day"


def test_daily_load_multi_component():
    """Each ComponentType accumulates separately; one exceeding threshold doesn't affect others."""
    from app.models.enums import ComponentType
    from app.services.trigger_service import _compute_daily_loads

    meals = [
        MockMeal(
            timestamp=_make_utc(2026, 7, 5),
            items=[
                MockItem(components=[
                    MockComponent(ComponentType.HISTAMINES, 70.0),
                    MockComponent(ComponentType.FODMAP, 30.0),
                ]),
            ],
        ),
    ]

    daily_loads, high_load_days = _compute_daily_loads(meals, high_load_threshold=60.0)

    day_key = next(iter(daily_loads))
    assert daily_loads[day_key][ComponentType.HISTAMINES] == 70.0
    assert daily_loads[day_key][ComponentType.FODMAP] == 30.0

    assert ComponentType.HISTAMINES in high_load_days, "HISTAMINES should be high-load"
    assert ComponentType.FODMAP not in high_load_days, "FODMAP (30) should not be high-load"


# ── Tests: apply_preparation_modifiers ───────────────────────────────────────


def test_preparation_modifier_applies_delta():
    """Sourdough fermentation should reduce fodmap_fructans by 40 points."""
    from app.services.food_ingestion import apply_preparation_modifiers

    base_scores = {"fodmap_fructans": 70.0}
    kb_food = {
        "preparation_modifiers": {
            "sourdough_fermented": {"fodmap_fructans": -40}
        }
    }
    result = apply_preparation_modifiers(base_scores, "sourdough", kb_food)

    assert result["fodmap_fructans"] == 30.0, (
        f"Expected 30.0 after sourdough modifier, got {result['fodmap_fructans']}"
    )


def test_preparation_modifier_clamped_at_zero():
    """A large negative delta should never push a score below 0."""
    from app.services.food_ingestion import apply_preparation_modifiers

    base_scores = {"fodmap_fructans": 20.0}
    kb_food = {
        "preparation_modifiers": {
            "sourdough_fermented": {"fodmap_fructans": -80}
        }
    }
    result = apply_preparation_modifiers(base_scores, "sourdough", kb_food)

    assert result["fodmap_fructans"] == 0.0, (
        f"Score clamped at 0.0, but got {result['fodmap_fructans']}"
    )


def test_preparation_modifier_clamped_at_hundred():
    """A positive delta should never push a score above 100."""
    from app.services.food_ingestion import apply_preparation_modifiers

    base_scores = {"histamine": 90.0}
    kb_food = {
        "preparation_modifiers": {
            "aged": {"histamine": 30}
        }
    }
    result = apply_preparation_modifiers(base_scores, "aged cheese", kb_food)

    assert result["histamine"] == 100.0, (
        f"Score clamped at 100.0, but got {result['histamine']}"
    )


def test_preparation_modifier_no_modifiers_in_kb():
    """Foods without preparation_modifiers return base scores unchanged."""
    from app.services.food_ingestion import apply_preparation_modifiers

    base_scores = {"gluten": 50.0, "fodmap_fructans": 30.0}
    kb_food: dict = {}  # no preparation_modifiers key
    result = apply_preparation_modifiers(base_scores, "sourdough", kb_food)

    assert result == base_scores


def test_preparation_modifier_none_prep_method():
    """None preparation_method returns base scores unchanged."""
    from app.services.food_ingestion import apply_preparation_modifiers

    base_scores = {"histamine": 60.0}
    kb_food = {
        "preparation_modifiers": {
            "fermented": {"histamine": 20}
        }
    }
    result = apply_preparation_modifiers(base_scores, None, kb_food)

    assert result == base_scores


# ── Tests: extract_preparation_method ────────────────────────────────────────


def test_extract_preparation_method_sourdough():
    from app.services.ai_orchestrator import extract_preparation_method

    base, prep = extract_preparation_method("sourdough bread")
    assert prep == "sourdough", f"Expected 'sourdough', got {prep!r}"
    assert "bread" in base, f"Expected base to contain 'bread', got {base!r}"


def test_extract_preparation_method_raw():
    from app.services.ai_orchestrator import extract_preparation_method

    base, prep = extract_preparation_method("raw spinach")
    assert prep == "raw", f"Expected 'raw', got {prep!r}"
    assert "spinach" in base, f"Expected base to contain 'spinach', got {base!r}"


def test_extract_preparation_method_no_keyword():
    from app.services.ai_orchestrator import extract_preparation_method

    base, prep = extract_preparation_method("chicken")
    assert prep is None, f"Expected None for plain food name, got {prep!r}"
    assert base == "chicken"


def test_extract_preparation_method_grilled():
    from app.services.ai_orchestrator import extract_preparation_method

    base, prep = extract_preparation_method("grilled salmon")
    assert prep == "grilled", f"Expected 'grilled', got {prep!r}"
    assert "salmon" in base


def test_extract_preparation_method_does_not_mutate():
    """Calling extract_preparation_method should not mutate the input string."""
    from app.services.ai_orchestrator import extract_preparation_method

    original = "roasted vegetables"
    _ = extract_preparation_method(original)
    assert original == "roasted vegetables"
