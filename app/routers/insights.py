"""Trigger insights endpoints — lag correlation, suspect foods, triggers."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.dependencies import get_current_user
from app.models.meal import Meal
from app.models.symptom import SymptomScore
from app.models.user import User
from app.schemas.insights import (
    LagCorrelationOut,
    LagCorrelationRow,
    SuspectFoodRow,
    SuspectFoodsOut,
)
from app.schemas.trigger import TriggerListOut, TriggerPredictionOut
from app.services import trigger_service
from app.services.assoc_guardrail import (
    BAYESIAN_TRIGGER_THRESHOLD,
    GuardrailResult,
    analyze_association_guardrail,
)
from app.services.case_crossover import analyze_case_crossover_triggers
from app.services.hierarchical_trigger import (
    ComponentTriggerResult,
    analyze_hierarchical_triggers,
    food_components_by_name,
)
from app.services.medication_service import (
    get_medicated_symptom_map,
    medication_adjusted_score,
)
from app.utils.confidence import evidence_confidence_label

# Sentinel score for a food that resolves to no scored KB component — it cannot be
# hierarchical-Bayes scored, so it sorts to the bottom rather than fabricating a value.
_UNSCORED = ComponentTriggerResult(
    component_type=None,  # type: ignore[arg-type]
    trigger_probability=0.0,
    score=0.0,
    ci_low=0.0,
    ci_high=0.0,
    beta=0.0,
    beta_se=0.0,
    n_obs=0,
    n_exposed=0,
    is_cold_start=True,
)


def _driver_for_food(
    food_name: str,
    name_comps: dict[str, set],
    by_comp: dict,
) -> ComponentTriggerResult:
    """Pick the highest-scoring hierarchical component this food carries.

    Attributes per-component posteriors back onto a logged food: a food's score is the
    max over the components it carries (join food name -> KB FoodComponentDetail).
    Returns the ``_UNSCORED`` sentinel when the food matches no scored component.
    """
    comps = name_comps.get((food_name or "").strip().lower(), set())
    candidates = [by_comp[c] for c in comps if c in by_comp]
    if not candidates:
        return _UNSCORED
    return max(candidates, key=lambda r: r.score)


def _guardrail_verdict(
    driver: ComponentTriggerResult,
    by_guard: dict,
) -> tuple[float | None, bool | None]:
    """(p_value, agreement) from the classical guardrail for a food's driver component.

    Agreement = whether the Bayesian flag (trigger_probability >= threshold) matches
    the guardrail's FDR-significance verdict. ``(None, None)`` when the component was
    not tested (degenerate 2x2) or has no guardrail result. This surfaces the "our
    Bayesian signal agrees with a classical association test" story per food.
    """
    guard: GuardrailResult | None = by_guard.get(driver.component_type)
    if guard is None or guard.test == "skipped" or guard.p_value is None:
        return None, None
    bayes_flag = driver.trigger_probability >= BAYESIAN_TRIGGER_THRESHOLD
    return round(guard.p_value, 8), (bayes_flag == guard.significant)


router = APIRouter(prefix="/api/v1/insights", tags=["insights"])


@router.get(
    "/triggers",
    response_model=TriggerListOut,
    summary="Get trigger predictions for the authenticated user",
)
async def get_triggers(
    status_filter: str | None = Query(
        None,
        alias="status",
        description="Filter by trigger status (suspect, probable, confirmed, cleared)",
    ),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TriggerListOut:
    triggers = await trigger_service.get_user_triggers(db, user.id, status_filter=status_filter)
    return TriggerListOut(
        user_id=user.id,
        triggers=[TriggerPredictionOut.model_validate(t) for t in triggers],
        total=len(triggers),
    )


@router.get(
    "/lag-correlation",
    response_model=LagCorrelationOut,
    summary="Symptom-food correlations across 24/48/72hr lag windows",
)
async def get_lag_correlation(
    lookback_days: int = Query(30, ge=7, le=180, description="Days of history to analyze"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LagCorrelationOut:
    """Return symptom-food correlations bucketed by 24h/48h/72h lag windows.

    For each (window, food, symptom) tuple, counts how many times the food appeared in
    a meal within the lag window before the symptom event (``sample_size``, a temporal
    co-occurrence view). The ``correlation_score`` is the hierarchical-Bayes association
    strength (0–100) of the food's driving component, so it is consistent with the
    suspect-foods leaderboard rather than a raw frequency. Only tuples with
    sample_size >= 2 are returned.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    windows = [24, 48, 72]
    rows: list[LagCorrelationRow] = []

    # Fetch all (non-deleted) symptoms in range
    symptom_result = await db.execute(
        select(SymptomScore)
        .where(
            and_(
                SymptomScore.user_id == user.id,
                SymptomScore.timestamp >= cutoff,
                SymptomScore.deleted_at.is_(None),
            )
        )
        .order_by(SymptomScore.timestamp)
    )
    symptoms = list(symptom_result.scalars().all())

    if not symptoms:
        return LagCorrelationOut(correlations=[], total=0)

    # Fetch all (non-deleted) meals with items in range (go back further to cover max window)
    meal_cutoff = cutoff - timedelta(hours=max(windows))
    meal_result = await db.execute(
        select(Meal)
        .where(
            and_(
                Meal.user_id == user.id,
                Meal.timestamp >= meal_cutoff,
                Meal.deleted_at.is_(None),
            )
        )
        .options(selectinload(Meal.items))
    )
    meals = list(meal_result.scalars().unique().all())

    # Which symptom episodes were medicated? (covariate — see medication_service)
    medicated_map = await get_medicated_symptom_map(
        db, user.id, [s.id for s in symptoms]
    )

    # Build correlation buckets: (window, food_name, symptom_type) -> {count, symptom_ids}
    buckets: dict[tuple[int, str, str], dict] = {}

    for symptom in symptoms:
        for window_hours in windows:
            window_start = symptom.timestamp - timedelta(hours=window_hours)
            for meal in meals:
                if window_start <= meal.timestamp <= symptom.timestamp:
                    for item in meal.items:
                        key = (window_hours, item.name, str(symptom.symptom_type))
                        bucket = buckets.setdefault(key, {"count": 0, "symptom_ids": set()})
                        bucket["count"] += 1
                        bucket["symptom_ids"].add(symptom.id)

    # Hierarchical component posteriors (computed once) + food->component attribution.
    hier = await analyze_hierarchical_triggers(db, user.id, lookback_days=lookback_days)
    by_comp = {r.component_type: r for r in hier}
    food_names = {food_name for (_w, food_name, _s) in buckets}
    name_comps = await food_components_by_name(db, food_names)

    for (window_hours, food_name, symptom_name), bucket in buckets.items():
        sample_size = bucket["count"]
        if sample_size >= 2:
            driver = _driver_for_food(food_name, name_comps, by_comp)
            n_medicated = sum(1 for sid in bucket["symptom_ids"] if sid in medicated_map)
            rows.append(
                LagCorrelationRow(
                    window_hours=window_hours,
                    food_name=food_name,
                    symptom_name=symptom_name,
                    correlation_score=round(driver.score, 2),
                    sample_size=sample_size,
                    n_medicated_episodes=n_medicated,
                    medication_confounded=n_medicated > 0,
                    method=driver.method,
                    trigger_probability=round(driver.trigger_probability, 4),
                )
            )

    # Deterministic tie-break (score, then food, then window) — see get_suspect_foods.
    rows.sort(key=lambda r: (-r.correlation_score, r.food_name, r.window_hours))
    return LagCorrelationOut(correlations=rows, total=len(rows))


@router.get(
    "/suspect-foods",
    response_model=SuspectFoodsOut,
    summary="Leaderboard of foods with highest trigger correlation",
)
async def get_suspect_foods(
    lookback_days: int = Query(30, ge=7, le=180, description="Days of history to analyze"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SuspectFoodsOut:
    """Return foods ranked by hierarchical-Bayes trigger association score.

    Only foods that preceded >= 3 distinct symptom episodes are included. Scoring is the
    hierarchical logistic engine: a food's score is the max over the components it
    carries (join food name -> KB FoodComponentDetail) of the per-component posterior.
    Every entry carries a medication-adjusted ``combined_score``, a 95% odds-ratio
    CREDIBLE interval (``ci_low``/``ci_high``), the ``n_meals`` / ``n_symptom_episodes``
    sample sizes, medication-covariate counts, a plain-English ``confidence_label``, the
    de-confounded ``trigger_probability``, and the frequentist FDR guardrail's
    ``assoc_p_value`` / ``assoc_agreement`` (the "hybrid" classical check).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Fetch (non-deleted) symptoms
    symptom_result = await db.execute(
        select(SymptomScore).where(
            and_(
                SymptomScore.user_id == user.id,
                SymptomScore.timestamp >= cutoff,
                SymptomScore.deleted_at.is_(None),
            )
        )
    )
    symptoms = list(symptom_result.scalars().all())

    if not symptoms:
        return SuspectFoodsOut(foods=[], total=0)

    # Fetch (non-deleted) meals with items, reaching back 72h before the window so a
    # meal just before the cutoff can still precede an early symptom.
    meal_result = await db.execute(
        select(Meal)
        .where(
            and_(
                Meal.user_id == user.id,
                Meal.timestamp >= cutoff - timedelta(hours=72),
                Meal.deleted_at.is_(None),
            )
        )
        .options(selectinload(Meal.items))
    )
    meals = list(meal_result.scalars().unique().all())

    # Which symptom episodes were medicated? (covariate — see medication_service)
    medicated_map = await get_medicated_symptom_map(
        db, user.id, [s.id for s in symptoms]
    )

    # Per food: distinct symptom episodes it preceded (within 72h) and their ids.
    food_episode_ids: dict[str, set] = {}
    for symptom in symptoms:
        window_start = symptom.timestamp - timedelta(hours=72)
        seen_foods: set[str] = set()
        for meal in meals:
            if window_start <= meal.timestamp <= symptom.timestamp:
                for item in meal.items:
                    seen_foods.add(item.name)
        for food_name in seen_foods:
            food_episode_ids.setdefault(food_name, set()).add(symptom.id)

    # Base-rate denominator: distinct meals in the lookback window containing each food.
    food_meal_counts: dict[str, int] = {}
    for meal in meals:
        if meal.timestamp < cutoff:
            continue  # only count meals inside the reported lookback window
        for food_name in {item.name for item in meal.items}:
            food_meal_counts[food_name] = food_meal_counts.get(food_name, 0) + 1

    # Hierarchical posteriors + classical FDR guardrail (computed once each) + the
    # food->component attribution needed to project them onto logged foods.
    hier = await analyze_hierarchical_triggers(db, user.id, lookback_days=lookback_days)
    by_comp = {r.component_type: r for r in hier}
    guardrail = await analyze_association_guardrail(db, user.id, lookback_days=lookback_days)
    by_guard = {g.component_type: g for g in guardrail}
    qualifying_names = {
        name for name, ids in food_episode_ids.items() if len(ids) >= 3
    }
    name_comps = await food_components_by_name(db, qualifying_names)

    # PRIMARY scorer: the within-person case-crossover (Phase 3 bake-off winner). It is
    # food-level, so unlike the component model it can exonerate an innocent food that
    # merely shares a trigger's component. The component model stays as the FALLBACK for
    # any food the case-crossover cannot test (no within-person exposure discordance),
    # and its per-component guardrail remains the corroborating agreement check.
    cc_by_food = {
        r.food_name: r
        for r in await analyze_case_crossover_triggers(
            db, user.id, lookback_days=lookback_days, candidate_foods=qualifying_names
        )
    }

    result_foods: list[SuspectFoodRow] = []

    for food_name, episode_ids in food_episode_ids.items():
        n_symptom_episodes = len(episode_ids)
        if n_symptom_episodes < 3:
            continue

        driver = _driver_for_food(food_name, name_comps, by_comp)  # component fallback + guardrail
        cc = cc_by_food.get(food_name)
        if cc is not None and cc.testable:
            trigger_score = cc.score               # 0–100 decoupled rank/flag score
            ci_low, ci_high = cc.ci_low, cc.ci_high  # odds-ratio 95% CI
            method = cc.method
            trigger_probability = cc.score / 100.0
        else:
            # Cold-start / untestable food -> per-component hierarchical driver.
            trigger_score = driver.score
            ci_low, ci_high = driver.ci_low, driver.ci_high
            method = driver.method
            trigger_probability = driver.trigger_probability

        n_medicated = sum(1 for sid in episode_ids if sid in medicated_map)
        combined_score = medication_adjusted_score(
            trigger_score, n_symptom_episodes, n_medicated
        )

        confidence_label = evidence_confidence_label(n_symptom_episodes, ci_low, ci_high)
        assoc_p, assoc_agree = _guardrail_verdict(driver, by_guard)

        # Honesty demotion: when the classical FDR guardrail actively DISAGREES with
        # the Bayesian flag (assoc_agreement is False — the guardrail tested this
        # component and did not corroborate), cap the label at "Preliminary". A
        # single-method claim we can't cross-validate is never surfaced as an
        # established signal. (None = guardrail had no verdict → no demotion.)
        if assoc_agree is False and confidence_label != "Preliminary":
            confidence_label = "Preliminary"

        result_foods.append(
            SuspectFoodRow(
                food_name=food_name,
                trigger_score=round(trigger_score, 2),
                combined_score=round(combined_score, 2),
                ci_low=round(ci_low, 2),
                ci_high=round(ci_high, 2),
                n_meals=food_meal_counts.get(food_name, 0),
                n_symptom_episodes=n_symptom_episodes,
                n_medicated_episodes=n_medicated,
                medication_confounded=n_medicated > 0,
                confidence_label=confidence_label,
                sample_size=n_symptom_episodes,
                method=method,
                trigger_probability=round(trigger_probability, 4),
                assoc_p_value=assoc_p,
                assoc_agreement=assoc_agree,
            )
        )

    # Break score ties by food name so the leaderboard order is deterministic (the raw
    # dict/set iteration order that feeds this is hash-seed dependent).
    result_foods.sort(key=lambda f: (-f.combined_score, f.food_name))
    return SuspectFoodsOut(foods=result_foods, total=len(result_foods))
