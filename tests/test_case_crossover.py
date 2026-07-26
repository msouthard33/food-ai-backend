"""Tests for the within-person case-crossover engine (Phase 3 bake-off winner).

Two layers:
  * Pure-core tests of ``score_case_crossover`` (no DB): a clean trigger is flagged, a
    ubiquitous safe food is not, the rank/flag decoupling invariant holds, deterministic.
  * A DB scenario: the DECOY case that motivated productionizing this engine — Garlic
    causes symptoms, Onion is eaten just as often on other days with NO symptom. The
    component model can't tell them apart; the case-crossover flags Garlic and
    exonerates Onion.
"""

import uuid
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import text

from app.models.enums import ConditionType, MealType, SymptomType
from app.models.food import FoodComponentDetail, FoodEntry
from app.models.meal import Meal, MealItem
from app.models.symptom import SymptomScore
from app.models.user import UserCondition
from app.services.case_crossover import (
    FDR_FLAG_Q,
    SUSPECT_FLOOR,
    analyze_case_crossover_triggers,
    score_case_crossover,
)
from tests.conftest import _ensure_tables, async_session_factory

_MONDAY = date(2026, 1, 5)  # a Monday, clean ISO-week boundary


def _build_diary(garlic_offsets, onion_offsets, symptom_offsets, weeks=6):
    """Return (days_food, symptom_days) over ``weeks`` weeks.

    Rice is eaten every day (a ubiquitous safe food). Garlic/Onion are eaten on the
    given weekday offsets each week; symptoms occur on ``symptom_offsets`` weekdays.
    """
    days_food: dict[date, set[str]] = {}
    symptom_days: set[date] = set()
    for w in range(weeks):
        for off in range(7):
            day = _MONDAY + timedelta(days=w * 7 + off)
            foods = {"Rice"}
            if off in garlic_offsets:
                foods.add("Garlic")
            if off in onion_offsets:
                foods.add("Onion")
            days_food[day] = foods
            if off in symptom_offsets:
                symptom_days.add(day)
    return days_food, symptom_days


def test_clean_trigger_is_flagged_safe_food_is_not():
    # Garlic on Mon/Fri -> symptom those days; Rice every day (no discordance).
    days_food, symptom_days = _build_diary(
        garlic_offsets={0, 4}, onion_offsets=set(), symptom_offsets={0, 4}
    )
    results = {r.food_name: r for r in score_case_crossover(days_food, symptom_days, {"Garlic", "Rice"})}
    garlic, rice = results["Garlic"], results["Rice"]

    assert garlic.odds_ratio > 1.0 and garlic.ci_low > 1.0
    assert garlic.flagged is True
    assert garlic.score >= SUSPECT_FLOOR
    assert garlic.testable

    # Rice is eaten on symptom and non-symptom days alike -> not a suspect.
    assert rice.flagged is False
    assert rice.score < SUSPECT_FLOOR


def test_decoupling_invariant_and_bounds():
    days_food, symptom_days = _build_diary(
        garlic_offsets={0, 4}, onion_offsets={2}, symptom_offsets={0, 4}
    )
    for r in score_case_crossover(days_food, symptom_days, {"Garlic", "Onion", "Rice"}):
        # A food is flagged IFF it clears the strict FDR gate; flag IFF score>=floor.
        assert r.flagged == (r.score >= SUSPECT_FLOOR)
        assert r.flagged == (r.odds_ratio > 1.0 and r.q_value <= FDR_FLAG_Q)
        assert 0.0 <= r.score <= 100.0


def test_deterministic_identical_across_calls():
    days_food, symptom_days = _build_diary(
        garlic_offsets={0, 4}, onion_offsets={2}, symptom_offsets={0, 4}
    )
    a = score_case_crossover(days_food, symptom_days, {"Garlic", "Onion", "Rice"})
    b = score_case_crossover(days_food, symptom_days, {"Garlic", "Onion", "Rice"})
    assert [(r.food_name, r.score, r.odds_ratio) for r in a] == [
        (r.food_name, r.score, r.odds_ratio) for r in b
    ]


def test_empty_inputs_are_untestable():
    results = score_case_crossover({}, set(), {"Garlic"})
    assert len(results) == 1
    assert results[0].testable is False
    assert results[0].score == 0.0


# ── DB scenario: the decoy exoneration (the reason this engine was promoted) ───

@pytest.mark.asyncio
async def test_decoy_food_is_exonerated_db():
    """Garlic causes symptoms; Onion (eaten just as often, different days) does not.

    Both carry FODMAP, so the per-component model gives them the same score and cannot
    exonerate Onion. The case-crossover, being food-level, must flag Garlic and NOT
    flag the innocent Onion.
    """
    await _ensure_tables()
    uid = uuid.uuid4()
    base = (datetime.now(UTC) - timedelta(weeks=6)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    async with async_session_factory() as session:
        await session.execute(
            text(
                "INSERT INTO users (id, email, timezone, onboarding_completed) "
                "VALUES (:id, :email, 'UTC', true)"
            ),
            {"id": uid, "email": f"{uid}@foodai.test"},
        )
        session.add(UserCondition(user_id=uid, condition_type=ConditionType.IBS))

        ids = {}
        for name in ("Garlic", "Onion", "Rice"):
            food = FoodEntry(name=name)
            session.add(food)
            await session.flush()
            ids[name] = food.id
        await session.flush()

        for dnum in range(6 * 7):
            day = base + timedelta(days=dnum)
            mod = dnum % 7
            # Rice every day at lunch; Garlic dinner Mon/Fri (+symptom 8h later),
            # Onion dinner Wed (NO symptom).
            for name, hour in [("Rice", 12)]:
                m = Meal(user_id=uid, timestamp=day + timedelta(hours=hour), meal_type=MealType.LUNCH)
                session.add(m); await session.flush()
                session.add(MealItem(meal_id=m.id, food_entry_id=ids[name], name=name))
            trigger = "Garlic" if mod in (0, 4) else ("Onion" if mod == 2 else None)
            if trigger:
                m = Meal(user_id=uid, timestamp=day + timedelta(hours=19), meal_type=MealType.DINNER)
                session.add(m); await session.flush()
                session.add(MealItem(meal_id=m.id, food_entry_id=ids[trigger], name=trigger))
                if trigger == "Garlic":
                    session.add(
                        SymptomScore(
                            user_id=uid,
                            timestamp=day + timedelta(hours=19 + 8),  # 8h IBS onset
                            symptom_type=SymptomType.BLOATING,
                            vas_score=70,
                        )
                    )
        await session.commit()

    async with async_session_factory() as session:
        results = {
            r.food_name: r
            for r in await analyze_case_crossover_triggers(
                session, uid, lookback_days=6 * 7 + 3
            )
        }

    garlic, onion = results["Garlic"], results["Onion"]
    assert garlic.flagged is True, f"Garlic not flagged: {garlic}"
    assert onion.flagged is False, f"innocent Onion was flagged: {onion}"
    assert garlic.score > onion.score
