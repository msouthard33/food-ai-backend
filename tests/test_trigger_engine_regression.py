"""Regression fixtures for the three demo accounts (trigger-engine honesty).

These lock in the trigger-engine correctness guarantees behind the live demo, framed
around the exact personas ``scripts/seed_demo_account.py`` seeds:

  * demo_ibs   — Garlic/Onion (FODMAP), 8h onset  -> "Garlic never reads protective"
                                                     + "true trigger ranks #1"
  * demo_mcas  — Cheese/Salmon (histamines), 3h    -> "MCAS intervals stay finite"

Each test seeds a compact but faithful 6-week diary through the ORM (mirroring the
demo generator's weekly trigger cadence and 8h/3h onset lags) and asserts against the
PRODUCTION engine (``analyze_hierarchical_triggers``), so a regression in either the
numerical guard (Phase 1) or the exposure/lag model (Phase 2) trips a red test.

Everything is deterministic: fixed food rotation, fixed weekday cadence, no RNG in the
assertions. Requires the local ``foodai_test`` Postgres (same as the other engine tests).
"""

import math
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
from app.services.hierarchical_trigger import (
    ATTRIBUTION_LEVEL_THRESHOLD,
    analyze_hierarchical_triggers,
)
from tests.conftest import _ensure_tables, async_session_factory

# A compact rotation of distinct "safe" staples (no scored component), wide enough
# that no single safe food blankets the symptom windows — mirroring the demo seed's
# intent that only the planted trigger surfaces.
_SAFE_FOODS = [
    "Reg White Rice", "Reg Chicken", "Reg Beef", "Reg Turkey", "Reg Pork",
    "Reg Lamb", "Reg Olive Oil", "Reg Butter", "Reg Coffee", "Reg Espresso",
    "Reg Green Tea", "Reg Duck", "Reg Cumin", "Reg Sunflower Oil",
]

_MEAL_SCHEDULE = [(MealType.BREAKFAST, 8, 0), (MealType.LUNCH, 12, 30), (MealType.DINNER, 19, 0)]
_PRIMARY_DAY_MODS = {1, 5}
_SECONDARY_DAY_MODS = {3}
_NUM_WEEKS = 6


async def _seed_demo_like_diary(
    condition: ConditionType,
    primary_food: str,
    secondary_food: str,
    trigger_component: ComponentType,
    symptom_type: SymptomType,
    lag_hours: float,
) -> uuid.UUID:
    """Seed one demo-like patient (6 weeks, 3 meals/day) and return the user id.

    Trigger cadence follows the demo generator: primary trigger on weekday-mods
    ``{1, 5}`` and secondary on ``{3}``, both at dinner, with a symptom ``lag_hours``
    later. Safe staples rotate two-per-meal so incidental foods stay below the signal.
    The trigger food carries ``trigger_component`` at level 4 (KB 0–4 scale, well above
    the attribution threshold); safe foods carry no scored component.
    """
    await _ensure_tables()
    uid = uuid.uuid4()
    base = (datetime.now(UTC) - timedelta(weeks=_NUM_WEEKS)).replace(
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
        session.add(UserCondition(user_id=uid, condition_type=condition))

        # Foods: triggers carry the component at level 4; safe foods carry nothing.
        food_ids: dict[str, uuid.UUID] = {}
        for name, comp in [
            (primary_food, trigger_component),
            (secondary_food, trigger_component),
            *[(s, None) for s in _SAFE_FOODS],
        ]:
            food = FoodEntry(name=name)
            session.add(food)
            await session.flush()
            if comp is not None:
                session.add(
                    FoodComponentDetail(
                        food_entry_id=food.id, component_type=comp, level=Decimal("4")
                    )
                )
            food_ids[name] = food.id
        await session.flush()

        safe_idx = 0

        def next_safe() -> str:
            nonlocal safe_idx
            name = _SAFE_FOODS[safe_idx % len(_SAFE_FOODS)]
            safe_idx += 1
            return name

        for day in range(_NUM_WEEKS * 7):
            day_base = base + timedelta(days=day)
            mod = day % 7
            for meal_type, hour, minute in _MEAL_SCHEDULE:
                ts = day_base + timedelta(hours=hour, minutes=minute)
                meal = Meal(user_id=uid, timestamp=ts, meal_type=meal_type)
                session.add(meal)
                await session.flush()

                names = [next_safe(), next_safe()]
                trigger_here: str | None = None
                if meal_type == MealType.DINNER:
                    if mod in _PRIMARY_DAY_MODS:
                        trigger_here = primary_food
                    elif mod in _SECONDARY_DAY_MODS:
                        trigger_here = secondary_food
                if trigger_here is not None:
                    names.append(trigger_here)

                for name in names:
                    session.add(
                        MealItem(meal_id=meal.id, food_entry_id=food_ids[name], name=name)
                    )
                await session.flush()

                if trigger_here is not None:
                    session.add(
                        SymptomScore(
                            user_id=uid,
                            timestamp=ts + timedelta(hours=lag_hours),
                            symptom_type=symptom_type,
                            vas_score=70,
                        )
                    )
        await session.commit()
    return uid


@pytest.mark.asyncio
async def test_mcas_demo_intervals_stay_finite():
    """MCAS demo: every component's odds-ratio interval is finite (Phase 1 guard).

    Cheese/Salmon (histamines) with a 3h onset produces near-perfect separation on the
    histamine family — exactly the quasi-separation that made the un-globalized Newton
    solver diverge and overflow the interval to ~1e37. The line-search guard must keep
    every ci_high finite and below the 1e12 finiteness ceiling.
    """
    uid = await _seed_demo_like_diary(
        ConditionType.MCAS,
        "Demo Cheese Cheddar",
        "Demo Salmon Smoked",
        ComponentType.HISTAMINES,
        SymptomType.SKIN_REACTION,
        lag_hours=3.0,
    )
    async with async_session_factory() as session:
        results = await analyze_hierarchical_triggers(session, uid, lookback_days=_NUM_WEEKS * 7 + 3)

    assert results, "expected at least the histamine-family components"
    for r in results:
        assert math.isfinite(r.ci_low), f"{r.component_type} ci_low not finite"
        assert math.isfinite(r.ci_high), f"{r.component_type} ci_high not finite"
        assert r.ci_high < 1e12, f"{r.component_type} ci_high={r.ci_high} overflowed"
        assert math.isfinite(r.beta) and abs(r.beta) <= 15.0 + 1e-9


@pytest.mark.asyncio
async def test_mcas_demo_histamines_is_a_signal_not_protective():
    """MCAS demo: the histamine component reads as a trigger, never protective.

    A finite interval is necessary but not sufficient — it must also point the right
    way. Histamines (the planted cause) must have an odds-ratio interval that is not
    entirely below 1.0.
    """
    uid = await _seed_demo_like_diary(
        ConditionType.MCAS,
        "Demo Cheese Cheddar 2",
        "Demo Salmon Smoked 2",
        ComponentType.HISTAMINES,
        SymptomType.SKIN_REACTION,
        lag_hours=3.0,
    )
    async with async_session_factory() as session:
        results = await analyze_hierarchical_triggers(session, uid, lookback_days=_NUM_WEEKS * 7 + 3)

    by_comp = {r.component_type: r for r in results}
    hist = by_comp[ComponentType.HISTAMINES]
    assert hist.ci_high > 1.0, f"histamines read protective: OR CI [{hist.ci_low}, {hist.ci_high}]"


# ── IBS demo (demo_ibs): Garlic/Onion, FODMAP, 8h onset ───────────────────────

async def _ibs_component_and_food_scores() -> tuple[dict, dict]:
    """Seed the IBS demo diary and return (component results, food->driver score).

    Projects the per-component hierarchical posteriors back onto logged foods exactly
    the way the /suspect-foods endpoint does (driver = the highest-scoring component a
    food carries at the attribution level), so "Garlic ranks #1" is tested against the
    same projection patients see.
    """
    uid = await _seed_demo_like_diary(
        ConditionType.IBS,
        "Demo Garlic",
        "Demo Onion",
        ComponentType.FODMAP,
        SymptomType.BLOATING,
        lag_hours=8.0,
    )
    async with async_session_factory() as session:
        results = await analyze_hierarchical_triggers(
            session, uid, lookback_days=_NUM_WEEKS * 7 + 3
        )
        # Food -> KB components at the attribution level (same helper the endpoint uses).
        from app.services.hierarchical_trigger import food_components_by_name

        food_names = {"Demo Garlic", "Demo Onion", *_SAFE_FOODS}
        name_comps = await food_components_by_name(session, food_names)

    by_comp = {r.component_type: r for r in results}
    food_scores: dict[str, float] = {}
    for name in food_names:
        comps = name_comps.get(name.lower(), set())
        drivers = [by_comp[c] for c in comps if c in by_comp]
        food_scores[name] = max((d.score for d in drivers), default=0.0)
    return by_comp, food_scores


@pytest.mark.asyncio
async def test_ibs_demo_garlic_never_reads_protective():
    """IBS demo: Garlic's driving component (FODMAP) is never protective.

    The headline honesty bug — a frequent trigger (garlic every other day) fell inside
    the old 36h max-lag window on nearly every day, leaving no control days, so the
    fitted FODMAP effect inverted and Garlic read *protective* (OR interval entirely
    below 1.0). With the onset-aligned exposure window the odds-ratio interval must sit
    on the trigger side: ci_high > 1.0 and the point estimate (β) positive.
    """
    by_comp, _ = await _ibs_component_and_food_scores()
    fodmap = by_comp[ComponentType.FODMAP]
    assert fodmap.beta > 0.0, f"FODMAP β negative (protective): {fodmap.beta}"
    assert fodmap.ci_high > 1.0, (
        f"FODMAP reads protective: OR CI [{fodmap.ci_low}, {fodmap.ci_high}]"
    )
    assert fodmap.ci_low > 1.0, (
        "a clean demo trigger should have an OR interval excluding 1.0: "
        f"[{fodmap.ci_low}, {fodmap.ci_high}]"
    )


@pytest.mark.asyncio
async def test_ibs_demo_true_trigger_ranks_first():
    """IBS demo: the planted trigger (Garlic) is the #1 suspect food.

    End-to-end acceptance for the demo: the highest-scoring food (via the endpoint's
    component->food projection) is a planted trigger, and it clears the strong-signal
    band while every safe staple stays well below it.
    """
    _, food_scores = await _ibs_component_and_food_scores()
    ranked = sorted(food_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_food, top_score = ranked[0]
    assert top_food in {"Demo Garlic", "Demo Onion"}, f"top suspect is {top_food}, not a trigger"
    assert top_score >= 85.0, f"top trigger score {top_score} below the strong-signal band"
    best_safe = max((food_scores[s] for s in _SAFE_FOODS), default=0.0)
    assert best_safe < top_score, "a safe food scored at/above the planted trigger"


# ── Histamine-intolerance demo (demo_histamine): Tuna/Feta, histamines, 4h ────

@pytest.mark.asyncio
async def test_histamine_demo_finite_and_directionally_correct():
    """Histamine-intolerance demo: histamines read as a finite, non-protective trigger.

    Third demo persona (Tuna/Feta, 4h onset). Same guarantees as the other two accounts:
    the odds-ratio interval is finite (Phase 1 guard) and points the trigger way
    (Phase 2 exposure), never protective.
    """
    uid = await _seed_demo_like_diary(
        ConditionType.HISTAMINE_INTOLERANCE,
        "Demo Tuna Canned",
        "Demo Feta Cheese",
        ComponentType.HISTAMINES,
        SymptomType.HEADACHE,
        lag_hours=4.0,
    )
    async with async_session_factory() as session:
        results = await analyze_hierarchical_triggers(
            session, uid, lookback_days=_NUM_WEEKS * 7 + 3
        )

    hist = {r.component_type: r for r in results}[ComponentType.HISTAMINES]
    assert math.isfinite(hist.ci_high) and hist.ci_high < 1e12
    assert hist.ci_high > 1.0, f"histamines protective: [{hist.ci_low}, {hist.ci_high}]"
