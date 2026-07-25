"""Tests for the Beta-Binomial Bayesian trigger engine.

Two layers:
  * Pure-math unit tests for the Beta helpers added to app/utils/confidence.py
    (no DB, no async).
  * Async engine scenarios that seed a small diary directly via the ORM and assert
    the de-confounding / cold-start / determinism behaviour of
    analyze_bayesian_triggers.
"""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import text

from app.models.enums import ComponentType, ConditionType, MealType, SymptomType
from app.models.food import FoodComponentDetail, FoodEntry
from app.models.meal import Meal, MealItem
from app.models.symptom import SymptomScore
from app.models.user import UserCondition
from app.services.bayesian_trigger import (
    DEFAULT_POPULATION_RATE,
    analyze_bayesian_triggers,
    prior_score,
)
from tests.conftest import _ensure_tables, async_session_factory

# ── Pure-math unit tests ──────────────────────────────────────────────────────

def test_beta_ppf_uniform_median():
    from app.utils.confidence import beta_ppf

    # Median of the uniform Beta(1,1) is exactly 0.5.
    assert abs(beta_ppf(0.5, 1, 1) - 0.5) < 1e-6
    # Clamps outside (0, 1).
    assert beta_ppf(0.0, 2, 5) == 0.0
    assert beta_ppf(1.0, 2, 5) == 1.0


def test_beta_ppf_monotonic_in_p():
    from app.utils.confidence import beta_ppf

    q10 = beta_ppf(0.10, 4, 6)
    q50 = beta_ppf(0.50, 4, 6)
    q90 = beta_ppf(0.90, 4, 6)
    assert q10 < q50 < q90


def test_beta_credible_interval_symmetric_when_a_equals_b():
    from app.utils.confidence import beta_credible_interval

    low, high = beta_credible_interval(5, 5)
    # Symmetric distribution -> interval symmetric about 0.5.
    assert abs(low - (1.0 - high)) < 1e-4
    assert low < 0.5 < high


def test_beta_credible_interval_narrows_with_more_data():
    from app.utils.confidence import beta_credible_interval

    wide_low, wide_high = beta_credible_interval(2, 2)
    tight_low, tight_high = beta_credible_interval(50, 50)
    assert (tight_high - tight_low) < (wide_high - wide_low)


def test_beta_mean():
    from app.utils.confidence import beta_mean

    assert abs(beta_mean(2, 8) - 0.2) < 1e-12
    assert abs(beta_mean(1, 1) - 0.5) < 1e-12


def test_prob_beta_exceeds_known_and_monotone():
    from app.utils.confidence import prob_beta_exceeds

    # Required known value: P(Beta(2,1) > Beta(1,2)) > 0.5.
    assert prob_beta_exceeds(2, 1, 1, 2) > 0.5
    # Identical distributions -> ~0.5.
    assert abs(prob_beta_exceeds(3, 3, 3, 3) - 0.5) < 0.02
    # Clear separation -> near 1 / near 0, and symmetric.
    high = prob_beta_exceeds(20, 2, 2, 20)
    low = prob_beta_exceeds(2, 20, 20, 2)
    assert high > 0.99
    assert low < 0.01
    # Monotone: shifting X's mass up only increases P(X > Y).
    assert prob_beta_exceeds(6, 4, 3, 7) > prob_beta_exceeds(4, 6, 3, 7)


# ── Engine test helpers ───────────────────────────────────────────────────────

async def _new_user(condition: ConditionType | None = None) -> uuid.UUID:
    """Insert a fresh non-synthetic user, optionally with one condition."""
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
        if condition is not None:
            session.add(UserCondition(user_id=uid, condition_type=condition))
        await session.commit()
    return uid


async def _make_food(session, name: str, component: ComponentType, level: float) -> uuid.UUID:
    """Create a FoodEntry carrying ``component`` at ``level`` (0–4)."""
    food = FoodEntry(name=name)
    session.add(food)
    await session.flush()
    session.add(
        FoodComponentDetail(
            food_entry_id=food.id,
            component_type=component,
            level=Decimal(str(level)),
        )
    )
    await session.flush()
    return food.id


async def _add_meal(session, uid: uuid.UUID, ts: datetime, food_id: uuid.UUID, name: str) -> None:
    meal = Meal(user_id=uid, timestamp=ts, meal_type=MealType.LUNCH)
    session.add(meal)
    await session.flush()
    session.add(MealItem(meal_id=meal.id, food_entry_id=food_id, name=name))
    await session.flush()


async def _add_symptom(session, uid: uuid.UUID, ts: datetime) -> None:
    session.add(
        SymptomScore(
            user_id=uid,
            timestamp=ts,
            symptom_type=SymptomType.BLOATING,
            vas_score=70,
        )
    )


def _day(days_ago: int, hour: int = 12) -> datetime:
    """A timestamp ``days_ago`` before now, at a fixed hour (stable calendar day)."""
    base = datetime.now(UTC) - timedelta(days=days_ago)
    return base.replace(hour=hour, minute=0, second=0, microsecond=0)


# ── Engine scenarios ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_strong_trigger_scores_high_with_tight_interval():
    """A component that ALWAYS precedes a symptom (and never appears on symptom-free
    days) -> high trigger probability, high score, CI excluding low values."""
    uid = await _new_user(ConditionType.IBS)
    async with async_session_factory() as session:
        fodmap_food = await _make_food(session, "Garlic X", ComponentType.FODMAP, 3.0)
        safe_food = await _make_food(session, "Rice X", ComponentType.OTHER, 0.0)

        # 28 days: even days eat FODMAP + get a symptom; odd days eat safe, no symptom.
        for d in range(28):
            if d % 2 == 0:
                await _add_meal(session, uid, _day(d, 9), fodmap_food, "Garlic X")
                await _add_symptom(session, uid, _day(d, 14))  # same-day, in window
            else:
                await _add_meal(session, uid, _day(d, 9), safe_food, "Rice X")
        await session.commit()

    async with async_session_factory() as session:
        results = await analyze_bayesian_triggers(session, uid, lookback_days=40)

    by_comp = {r.component_type: r for r in results}
    fodmap = by_comp[ComponentType.FODMAP]

    assert fodmap.a >= 13 and fodmap.b == 0  # every exposed day had a symptom
    assert fodmap.c == 0  # no symptoms on unexposed days
    assert fodmap.trigger_probability > 0.9
    assert fodmap.score > 90.0
    # Interval excludes low symptom rates. The lower bound sits ~58 (not higher)
    # because the prior contributes ~5 pseudo-failures that a mere 14 observations
    # do not fully overcome — intended conservative behaviour for a clinical value.
    assert fodmap.ci_low > 50.0
    assert fodmap.ci_low < fodmap.ci_high <= 100.0
    assert not fodmap.is_cold_start


@pytest.mark.asyncio
async def test_base_rate_food_is_deconfounded():
    """A food eaten often but whose symptoms are INDEPENDENT of exposure (same rate
    on exposed and unexposed days) -> low trigger probability, unlike a naive
    proportion score which would reward the frequent exposure."""
    uid = await _new_user()  # no condition -> no prior nudge
    async with async_session_factory() as session:
        food = await _make_food(session, "Bread X", ComponentType.FODMAP, 3.0)
        safe = await _make_food(session, "Rice Y", ComponentType.OTHER, 0.0)

        # 30 days. Exposure on even days (15). Symptoms on every 3rd day, which lands
        # on both exposed and unexposed days -> independent of exposure.
        for d in range(30):
            if d % 2 == 0:
                await _add_meal(session, uid, _day(d, 9), food, "Bread X")
            else:
                await _add_meal(session, uid, _day(d, 9), safe, "Rice Y")
            if d % 3 == 0:
                await _add_symptom(session, uid, _day(d, 14))
        await session.commit()

    async with async_session_factory() as session:
        results = await analyze_bayesian_triggers(session, uid, lookback_days=40)

    fodmap = {r.component_type: r for r in results}[ComponentType.FODMAP]

    # Symptoms split roughly evenly between exposed and unexposed days.
    assert fodmap.a > 0 and fodmap.c > 0
    # De-confounded: exposure does not raise the symptom rate -> near coin-flip, well
    # below the strong-trigger regime.
    assert fodmap.trigger_probability < 0.75
    assert fodmap.score < 75.0


@pytest.mark.asyncio
async def test_strong_beats_base_rate():
    """Same frequent food, but the strong-trigger user should out-score the
    base-rate user — the headline de-confounding guarantee."""
    strong_uid = await _new_user(ConditionType.IBS)
    base_uid = await _new_user()
    async with async_session_factory() as session:
        s_food = await _make_food(session, "Onion S", ComponentType.FRUCTOSE, 3.0)
        s_safe = await _make_food(session, "Oat S", ComponentType.OTHER, 0.0)
        b_food = await _make_food(session, "Onion B", ComponentType.FRUCTOSE, 3.0)
        b_safe = await _make_food(session, "Oat B", ComponentType.OTHER, 0.0)
        for d in range(28):
            if d % 2 == 0:
                await _add_meal(session, strong_uid, _day(d, 9), s_food, "Onion S")
                await _add_symptom(session, strong_uid, _day(d, 14))
                await _add_meal(session, base_uid, _day(d, 9), b_food, "Onion B")
            else:
                await _add_meal(session, strong_uid, _day(d, 9), s_safe, "Oat S")
                await _add_meal(session, base_uid, _day(d, 9), b_safe, "Oat B")
            if d % 3 == 0:
                await _add_symptom(session, base_uid, _day(d, 14))
        await session.commit()

    async with async_session_factory() as session:
        strong = await analyze_bayesian_triggers(session, strong_uid, lookback_days=40)
        base = await analyze_bayesian_triggers(session, base_uid, lookback_days=40)

    s = {r.component_type: r for r in strong}[ComponentType.FRUCTOSE]
    b = {r.component_type: r for r in base}[ComponentType.FRUCTOSE]
    assert s.score > b.score
    assert s.trigger_probability > 0.9 > b.trigger_probability


@pytest.mark.asyncio
async def test_cold_start_returns_condition_prior():
    """No diary data + an IBS condition -> the implicated components (FODMAP,
    LACTOSE, FRUCTOSE) come back elevated from the prior, not zero."""
    uid = await _new_user(ConditionType.IBS)
    async with async_session_factory() as session:
        results = await analyze_bayesian_triggers(session, uid, lookback_days=40)

    by_comp = {r.component_type: r for r in results}
    # Exactly the IBS-implicated components are candidates on cold start.
    assert ComponentType.FODMAP in by_comp
    assert ComponentType.LACTOSE in by_comp
    assert ComponentType.FRUCTOSE in by_comp

    fodmap = by_comp[ComponentType.FODMAP]
    assert fodmap.is_cold_start
    assert fodmap.a == fodmap.b == fodmap.c == fodmap.d == 0
    # Elevated relative to a non-implicated component's prior, and non-trivial.
    baseline = prior_score(DEFAULT_POPULATION_RATE, implicated=False)
    assert fodmap.score > baseline
    assert fodmap.score > 40.0


@pytest.mark.asyncio
async def test_determinism_identical_across_runs():
    """Same DB state -> byte-identical results (no Monte-Carlo, no ordering drift)."""
    uid = await _new_user(ConditionType.IBS)
    async with async_session_factory() as session:
        food = await _make_food(session, "Apple D", ComponentType.FODMAP, 3.0)
        safe = await _make_food(session, "Rice D", ComponentType.OTHER, 0.0)
        for d in range(20):
            if d % 2 == 0:
                await _add_meal(session, uid, _day(d, 9), food, "Apple D")
                if d % 4 == 0:
                    await _add_symptom(session, uid, _day(d, 14))
            else:
                await _add_meal(session, uid, _day(d, 9), safe, "Rice D")
        await session.commit()

    async with async_session_factory() as session:
        run1 = await analyze_bayesian_triggers(session, uid, lookback_days=40)
    async with async_session_factory() as session:
        run2 = await analyze_bayesian_triggers(session, uid, lookback_days=40)

    assert [r.to_dict() for r in run1] == [r.to_dict() for r in run2]
