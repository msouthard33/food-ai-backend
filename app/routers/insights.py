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
from app.services.bayesian_trigger import (
    ComponentTriggerResult,
    analyze_bayesian_triggers,
    food_components_by_name,
)
from app.services.medication_service import (
    get_medicated_symptom_map,
    medication_adjusted_score,
)
from app.utils.confidence import evidence_confidence_label

# Sentinel score for a food that resolves to no scored KB component — it cannot be
# Bayesian-scored, so it sorts to the bottom rather than fabricating a proportion.
_UNSCORED = ComponentTriggerResult(
    component_type=None,  # type: ignore[arg-type]
    trigger_probability=0.0,
    score=0.0,
    ci_low=0.0,
    ci_high=0.0,
    a=0, b=0, c=0, d=0,
    n_exposed_days=0,
    n_symptom_days=0,
    alpha_post=0.0,
    beta_post=0.0,
    alpha_unexposed_post=0.0,
    beta_unexposed_post=0.0,
    prior_alpha=0.0,
    prior_beta=0.0,
    is_cold_start=True,
)


def _driver_for_food(
    food_name: str,
    name_comps: dict,
    by_comp: dict,
) -> ComponentTriggerResult:
    """Pick the highest-scoring Bayesian component this food carries.

    Attributes per-component posteriors back onto a logged food: a food's score is
    the max over the components it carries (join food name -> KB FoodComponentDetail).
    Returns the ``_UNSCORED`` sentinel when the food matches no scored component.
    """
    comps = name_comps.get((food_name or "").strip().lower(), set())
    candidates = [by_comp[c] for c in comps if c in by_comp]
    if not candidates:
        return _UNSCORED
    return max(candidates, key=lambda r: r.score)

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

    For each (window, food, symptom) tuple, counts how many times the food
    appeared in a meal within the lag window before the symptom event (``sample_size``,
    a temporal co-occurrence view). The ``correlation_score`` is the Bayesian
    Beta-Binomial association strength (0–100) of the food's driving component, so it
    is consistent with the suspect-foods leaderboard rather than a raw frequency.
    Only tuples with sample_size >= 2 are returned.
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

    # Bayesian component posteriors (computed once) + food->component attribution.
    bayes = await analyze_bayesian_triggers(db, user.id, lookback_days=lookback_days)
    by_comp = {r.component_type: r for r in bayes}
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
                    trigger_probability=round(driver.trigger_probability, 4),
                )
            )

    rows.sort(key=lambda r: r.correlation_score, reverse=True)
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
    """Return foods ranked by Bayesian trigger association score.

    Only foods that preceded >= 3 distinct symptom episodes are included. Scoring is
    the Beta-Binomial engine: a food's score is the max over the components it carries
    (join food name -> KB FoodComponentDetail) of the per-component posterior. Every
    entry carries a medication-adjusted ``combined_score``, a 95% Bayesian CREDIBLE
    interval (``ci_low``/``ci_high``), the ``n_meals`` / ``n_symptom_episodes`` sample
    sizes, medication-covariate counts, a plain-English ``confidence_label``, and the
    de-confounded ``trigger_probability`` (Day-One Value honest-confidence doctrine).
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

    # Bayesian component posteriors (computed once) + food->component attribution.
    bayes = await analyze_bayesian_triggers(db, user.id, lookback_days=lookback_days)
    by_comp = {r.component_type: r for r in bayes}
    qualifying_names = {
        name for name, ids in food_episode_ids.items() if len(ids) >= 3
    }
    name_comps = await food_components_by_name(db, qualifying_names)

    result_foods: list[SuspectFoodRow] = []

    for food_name, episode_ids in food_episode_ids.items():
        n_symptom_episodes = len(episode_ids)
        if n_symptom_episodes < 3:
            continue

        # Score = the food's driving Bayesian component (max over components carried).
        driver = _driver_for_food(food_name, name_comps, by_comp)
        trigger_score = driver.score  # pre-medication Bayesian score (back-compat)
        ci_low, ci_high = driver.ci_low, driver.ci_high

        n_medicated = sum(1 for sid in episode_ids if sid in medicated_map)
        combined_score = medication_adjusted_score(
            trigger_score, n_symptom_episodes, n_medicated
        )

        confidence_label = evidence_confidence_label(n_symptom_episodes, ci_high - ci_low)

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
                trigger_probability=round(driver.trigger_probability, 4),
            )
        )

    result_foods.sort(key=lambda f: f.combined_score, reverse=True)
    return SuspectFoodsOut(foods=result_foods, total=len(result_foods))
