"""Trigger prediction engine — Beta-Binomial Bayesian scoring.

The scoring math lives in ``app.services.bayesian_trigger`` (the wired engine). This
module is the persistence + orchestration layer: it runs the engine per user, writes
the per-component posteriors onto ``TriggerPrediction`` rows, and seeds onboarding
cold-start priors. It no longer contains the legacy frequentist additive-score
(``calculate_confidence``) or per-meal/daily-load correlation builders — those were
replaced by the Beta-Binomial posterior (Bayesian Sprint 3).
"""

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import ComponentType, TriggerStatus
from app.models.meal import Meal
from app.models.trigger import TriggerPrediction

# NOTE: ``app.services.bayesian_trigger`` imports ``CONDITION_PRIORS`` from this module,
# so the Bayesian engine functions are imported LAZILY inside the functions below to
# avoid a circular import at module load.
if TYPE_CHECKING:
    from app.services.bayesian_trigger import ComponentTriggerResult

logger = logging.getLogger(__name__)

# Condition → component mapping for cold-start prior seeding.
# Keys are lowercase condition strings accepted by the onboarding endpoint.
CONDITION_PRIORS: dict[str, list[ComponentType]] = {
    "ibs": [ComponentType.FODMAP, ComponentType.LACTOSE, ComponentType.FRUCTOSE],
    "mcas": [ComponentType.HISTAMINES, ComponentType.SALICYLATES, ComponentType.OXALATES],
    "histamine_intolerance": [ComponentType.HISTAMINES],
    "food_allergy": [ComponentType.GLUTEN, ComponentType.MILK_DAIRY, ComponentType.EGGS],
}


async def get_user_triggers(
    db: AsyncSession,
    user_id: uuid.UUID,
    status_filter: str | None = None,
) -> list[TriggerPrediction]:
    """Fetch all trigger predictions for a user, optionally filtered by status."""
    q = select(TriggerPrediction).where(TriggerPrediction.user_id == user_id)
    if status_filter:
        q = q.where(TriggerPrediction.status == status_filter)
    q = q.order_by(TriggerPrediction.confidence_score.desc())
    result = await db.execute(q)
    return list(result.scalars().all())


def _compute_daily_loads(
    meals: list,
    high_load_threshold: float = 60.0,
) -> tuple[dict, dict]:
    """Pure helper: aggregate component loads by calendar day and identify high-load days.

    Retained daily-load aggregation utility (unit-tested without a DB session). The
    Beta-Binomial engine now owns trigger scoring; this helper is kept for the
    cumulative-dose ("bucket theory") daily-load view and its direct tests.

    Args:
        meals: ORM Meal objects (or any duck-typed objects with .timestamp, .items,
               item.components, and component.component_type / .estimated_level).
        high_load_threshold: Daily sum >= this value marks the day as high-load.

    Returns:
        (daily_loads, high_load_days) where:
          - daily_loads:   {date: {ComponentType: float}}
          - high_load_days: {ComponentType: [date, ...]}
    """
    daily_loads: dict = defaultdict(lambda: defaultdict(float))
    for meal in meals:
        meal_date = meal.timestamp.date()
        for item in meal.items:
            for component in item.components:
                daily_loads[meal_date][component.component_type] += float(
                    component.estimated_level or 0
                )

    high_load_days: dict = defaultdict(list)
    for day, comp_loads in daily_loads.items():
        for comp_type, load in comp_loads.items():
            if load >= high_load_threshold:
                high_load_days[comp_type].append(day)

    return dict(daily_loads), dict(high_load_days)


async def update_trigger_predictions(
    db: AsyncSession,
    user_id: uuid.UUID,
    results: list["ComponentTriggerResult"],
) -> list[TriggerPrediction]:
    """Upsert TriggerPrediction rows from Beta-Binomial engine results.

    ``confidence_score`` keeps its name and 0–100 scale but is now the component's
    Bayesian score (``trigger_probability * 100``). The raw posterior (alpha/beta),
    the 95% credible interval, and ``trigger_probability`` are persisted alongside so
    the score is auditable/reproducible and the clinician PDF can render the interval.

    Status mapping (thresholds unchanged, now applied to the Bayesian score):
      - < 15  -> CLEARED (only if a row already exists)
      - < 20  -> skipped (not surfaced)
      - 20–49 -> SUSPECT, 50–74 -> PROBABLE, 75+ -> CONFIRMED

    ``evidence_count`` = number of exposed days observed (0 on a cold-start prior row).
    """
    updated: list[TriggerPrediction] = []
    now = datetime.now(timezone.utc)

    for r in results:
        confidence = float(r.score)

        existing = await db.execute(
            select(TriggerPrediction).where(
                and_(
                    TriggerPrediction.user_id == user_id,
                    TriggerPrediction.component_type == r.component_type,
                )
            )
        )
        prediction = existing.scalar_one_or_none()

        if confidence < 15:
            if prediction is not None:
                prediction.status = TriggerStatus.CLEARED
                prediction.last_updated = now
            continue
        if confidence < 20:
            continue

        if confidence >= 75:
            status = TriggerStatus.CONFIRMED
        elif confidence >= 50:
            status = TriggerStatus.PROBABLE
        else:
            status = TriggerStatus.SUSPECT

        if prediction is None:
            prediction = TriggerPrediction(
                user_id=user_id,
                component_type=r.component_type,
                first_detected=now,
            )
            db.add(prediction)

        prediction.confidence_score = Decimal(str(round(confidence)))
        prediction.status = status
        prediction.last_updated = now
        prediction.evidence_count = r.n_exposed_days
        prediction.method = "bayesian_beta_binomial"
        prediction.trigger_probability = Decimal(str(round(r.trigger_probability, 5)))
        prediction.bayesian_alpha = Decimal(str(round(r.alpha_post, 4)))
        prediction.bayesian_beta = Decimal(str(round(r.beta_post, 4)))
        prediction.bayesian_ci_low = Decimal(str(round(r.ci_low, 2)))
        prediction.bayesian_ci_high = Decimal(str(round(r.ci_high, 2)))

        await db.flush()
        updated.append(prediction)

    await db.commit()
    logger.info(
        f"Updated {len(updated)} Bayesian trigger predictions for user {user_id}"
    )
    return updated


async def run_full_analysis(
    db: AsyncSession,
    user_id: uuid.UUID,
    lookback_days: int = 30,
) -> list[TriggerPrediction]:
    """Run the Beta-Binomial engine for a user and persist the results.

    Replaces the former frequentist pipeline (per-meal + daily-load correlation ->
    additive ``calculate_confidence``). The engine blends the condition/cohort PRIOR
    with observed data continuously, so a cold-start user surfaces their
    condition-implicated priors and a heavy logger becomes fully data-driven — the
    prior→data transition is smooth, with no ``SYNTHETIC_DECAY_THRESHOLD`` hard switch.
    """
    from app.services.bayesian_trigger import analyze_bayesian_triggers

    results = await analyze_bayesian_triggers(db, user_id, lookback_days=lookback_days)
    if not results:
        logger.info(f"No Bayesian trigger results for user {user_id}")
        return []

    predictions = await update_trigger_predictions(db, user_id, results)
    logger.info(
        f"Completed Bayesian trigger analysis for user {user_id}: "
        f"{len(predictions)} predictions"
    )
    return predictions


# ── Synthetic data decay rule ─────────────────────────────────────────────────

SYNTHETIC_DECAY_THRESHOLD = 42
"""DEPRECATED (Bayesian Sprint 3) — retained only as a documented constant.

This was the meal count at which population priors would be HARD-switched off for a
user. The Beta-Binomial engine makes that switch unnecessary: the posterior is
``prior + observed data``, so prior influence decays *continuously* as real
observations accumulate (a handful of exposed/unexposed days already dominate the
~6 pseudo-observation prior). There is no longer any hard cutover, and neither
``seed_condition_priors`` nor ``run_full_analysis`` reads this value.

42 meals ≈ 14 days of consistent 3-meals-per-day logging — kept for historical
reference (and back-compat with older tooling that imports the name).
"""


async def get_real_meal_count(db: AsyncSession, user_id: uuid.UUID) -> int:
    """Count real (non-synthetic) meal entries for a user.

    Used by the decay rule: once this count exceeds SYNTHETIC_DECAY_THRESHOLD,
    population-prior trigger rows should be excluded from insights.

    Note: synthetic patients live in separate user rows (is_synthetic=True),
    so all meals for a real user (is_synthetic=False) are by definition real.
    This query is a straightforward meal count for the given user_id.
    """
    from sqlalchemy import func

    result = await db.execute(
        select(func.count()).select_from(Meal).where(Meal.user_id == user_id)
    )
    return result.scalar_one()


async def seed_condition_priors(
    db: AsyncSession,
    user_id: uuid.UUID,
    condition_types: list[str],
) -> list[TriggerPrediction]:
    """Seed onboarding TriggerPrediction rows from the Beta-Binomial PRIOR.

    Cold start = the prior IS the posterior (zero observed data). For each
    ComponentType implicated by a declared condition (``CONDITION_PRIORS``) that the
    user does not already have a row for, we compute the engine's cold-start result
    (condition + cohort prior) and persist it through ``update_trigger_predictions`` —
    the same path the data-driven analysis uses. As the user logs meals and symptoms,
    ``run_full_analysis`` re-scores the same components and the posterior slides off the
    prior continuously; there is no hard ``SYNTHETIC_DECAY_THRESHOLD`` cutover.

    This replaces the old fixed ``confidence_score = 18`` cold-start hack: seeded rows
    now carry the actual Bayesian prior score, credible interval, and posterior params,
    so ``/insights/triggers`` reflects the real prior at zero data.

    Idempotent: components that already have a row are skipped, so re-calling with the
    same conditions creates 0 new rows.

    Args:
        db: Async database session
        user_id: UUID of the onboarding user
        condition_types: List of condition strings (e.g. ["ibs", "mcas"])

    Returns:
        List of newly created TriggerPrediction rows.
    """
    from app.services.bayesian_trigger import (
        DEFAULT_POPULATION_RATE,
        build_population_prior_table,
        cold_start_component_result,
    )

    implicated: set[ComponentType] = set()
    for condition in condition_types:
        implicated.update(CONDITION_PRIORS.get(condition.lower().strip(), []))

    if not implicated:
        logger.info(
            f"seed_condition_priors: no known conditions in {condition_types!r} "
            f"for user {user_id}"
        )
        return []

    existing = (
        await db.execute(
            select(TriggerPrediction.component_type).where(
                TriggerPrediction.user_id == user_id
            )
        )
    ).scalars().all()
    to_seed = implicated - set(existing)
    if not to_seed:
        logger.info(
            f"seed_condition_priors: all implicated components already seeded "
            f"for user {user_id}"
        )
        return []

    population_prior = await build_population_prior_table(db)
    results = [
        cold_start_component_result(
            comp,
            population_prior.get(comp, DEFAULT_POPULATION_RATE),
            implicated=True,
        )
        for comp in to_seed
    ]

    created = await update_trigger_predictions(db, user_id, results)
    logger.info(
        f"seed_condition_priors: {len(created)} prior rows created for user {user_id}"
    )
    return created
