"""Trigger prediction engine — hierarchical Bayesian scoring + classical guardrail.

The scoring math lives in ``app.services.hierarchical_trigger`` (the wired engine: a
joint per-user logistic regression with partial pooling) with the frequentist FDR
association guardrail in ``app.services.assoc_guardrail`` running alongside it (the
"hybrid": Bayesian signal + classical agreement check). This module is the
persistence + orchestration layer: it runs both per user, writes the per-component
posterior + guardrail verdict onto ``TriggerPrediction`` rows, and seeds onboarding
cold-start priors. It no longer contains the legacy frequentist additive-score
(``calculate_confidence``) or per-meal/daily-load correlation builders — those were
replaced by the hierarchical posterior (Sprint H4).
"""

import logging
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import ComponentType, TriggerStatus
from app.models.meal import Meal
from app.models.trigger import TriggerPrediction

# NOTE: ``hierarchical_trigger`` / ``bayesian_trigger`` / ``assoc_guardrail`` import
# ``CONDITION_PRIORS`` from this module, so those engine functions are imported LAZILY
# inside the functions below to avoid a circular import at module load.
if TYPE_CHECKING:
    from app.services.assoc_guardrail import GuardrailResult
    from app.services.hierarchical_trigger import ComponentTriggerResult

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
    hierarchical engine now owns trigger scoring; this helper is kept for the
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
    guardrail_by_comp: dict[ComponentType, "GuardrailResult"] | None = None,
) -> list[TriggerPrediction]:
    """Upsert TriggerPrediction rows from hierarchical-Bayes engine results.

    ``confidence_score`` keeps its name and 0–100 scale but is now the component's
    hierarchical score (``trigger_probability * 100``). The Laplace posterior (β, SE),
    the 95% odds-ratio credible interval, and ``trigger_probability`` are persisted
    alongside so the score is auditable/reproducible and the clinician PDF can render
    the interval. When ``guardrail_by_comp`` is supplied, each component's classical
    FDR guardrail p-value and its per-component agreement with the Bayesian flag are
    stored too (the "hybrid" classical check).

    Status mapping (thresholds unchanged, now applied to the hierarchical score):
      - < 15  -> CLEARED (only if a row already exists)
      - < 20  -> skipped (not surfaced)
      - 20–49 -> SUSPECT, 50–74 -> PROBABLE, 75+ -> CONFIRMED

    ``evidence_count`` = number of exposed days observed (0 on a cold-start prior row).
    """
    from app.services.assoc_guardrail import BAYESIAN_TRIGGER_THRESHOLD

    guardrail_by_comp = guardrail_by_comp or {}
    updated: list[TriggerPrediction] = []
    now = datetime.now(UTC)

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
                # Empty (not NULL) so mobile clients that decode a non-optional array
                # still deserialize; the joint engine does not attribute per-symptom.
                symptom_types=[],
            )
            db.add(prediction)

        # Per-component classical guardrail verdict (the hybrid agreement check).
        guard = guardrail_by_comp.get(r.component_type)
        if guard is not None and guard.test != "skipped" and guard.p_value is not None:
            assoc_p = Decimal(str(round(min(max(guard.p_value, 0.0), 1.0), 8)))
            bayes_flag = r.trigger_probability >= BAYESIAN_TRIGGER_THRESHOLD
            assoc_agreement: bool | None = bayes_flag == guard.significant
        else:
            assoc_p = None
            assoc_agreement = None

        prediction.confidence_score = Decimal(str(round(confidence)))
        prediction.status = status
        prediction.last_updated = now
        prediction.evidence_count = r.n_exposed
        prediction.method = r.method
        prediction.trigger_probability = Decimal(
            str(round(r.trigger_probability, 5))
        )
        prediction.bayes_beta = Decimal(str(round(r.beta, 6)))
        prediction.bayes_beta_se = Decimal(str(round(r.beta_se, 6)))
        prediction.bayes_ci_low = _safe_decimal(r.ci_low)
        prediction.bayes_ci_high = _safe_decimal(r.ci_high)
        prediction.assoc_p_value = assoc_p
        prediction.assoc_agreement = assoc_agreement

        await db.flush()
        updated.append(prediction)

    await db.commit()
    logger.info(
        f"Updated {len(updated)} hierarchical trigger predictions for user {user_id}"
    )
    return updated


def _safe_decimal(value: float, max_abs: float = 1.0e12) -> Decimal:
    """Round an odds-ratio interval bound and clamp it into the persisted column range.

    The credible interval is on the odds-ratio scale (``exp(β ± 1.96·SE)``) and is
    strictly positive but unbounded above; clamp so an extreme (near-separation) fit
    never overflows ``NUMERIC(18, 6)``.
    """
    clamped = min(max(value, 0.0), max_abs)
    return Decimal(str(round(clamped, 6)))


async def run_full_analysis(
    db: AsyncSession,
    user_id: uuid.UUID,
    lookback_days: int = 30,
) -> list[TriggerPrediction]:
    """Run the hierarchical engine + classical guardrail for a user and persist them.

    Replaces the former frequentist pipeline (per-meal + daily-load correlation ->
    additive ``calculate_confidence``). The hierarchical engine blends the
    condition/cohort PRIOR with observed data continuously, so a cold-start user
    surfaces their condition-implicated priors (implicated components elevated) and a
    heavy logger becomes fully data-driven — the prior→data transition is smooth, with
    no ``SYNTHETIC_DECAY_THRESHOLD`` hard switch. The frequentist FDR guardrail runs on
    the SAME lookback and its per-component p-value + agreement are persisted alongside
    (the "hybrid": Bayesian signal + classical agreement check).
    """
    from app.services.assoc_guardrail import analyze_association_guardrail
    from app.services.hierarchical_trigger import analyze_hierarchical_triggers

    results = await analyze_hierarchical_triggers(db, user_id, lookback_days=lookback_days)
    if not results:
        logger.info(f"No hierarchical trigger results for user {user_id}")
        return []

    guardrail = await analyze_association_guardrail(
        db, user_id, lookback_days=lookback_days
    )
    guardrail_by_comp = {g.component_type: g for g in guardrail}

    predictions = await update_trigger_predictions(
        db, user_id, results, guardrail_by_comp
    )
    logger.info(
        f"Completed hierarchical trigger analysis for user {user_id}: "
        f"{len(predictions)} predictions"
    )
    return predictions


# ── Synthetic data decay rule ─────────────────────────────────────────────────

SYNTHETIC_DECAY_THRESHOLD = 42
"""DEPRECATED (Sprint H4) — retained only as a documented constant.

This was the meal count at which population priors would be HARD-switched off for a
user. The hierarchical Bayesian engine makes that switch unnecessary: the MAP is a
ridge-to-prior penalized fit, so prior influence decays *continuously* as real
observations accumulate (a data-rich user overrides the prior; a small-n user is
shrunk toward it). There is no longer any hard cutover, and neither
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
    """Seed onboarding TriggerPrediction rows from the hierarchical-Bayes PRIOR.

    Cold start = the prior IS the posterior (zero observed data). For each
    ComponentType implicated by a declared condition (``CONDITION_PRIORS``) that the
    user does not already have a row for, we compute the engine's cold-start result
    (condition seed + cohort prior) and persist it through ``update_trigger_predictions``
    — the same path the data-driven analysis uses. As the user logs meals and symptoms,
    ``run_full_analysis`` re-scores the same components and the posterior slides off the
    prior continuously; there is no hard ``SYNTHETIC_DECAY_THRESHOLD`` cutover.

    This replaces the old fixed ``confidence_score = 18`` cold-start hack: seeded rows
    now carry the actual Bayesian prior score, odds-ratio credible interval, and
    posterior params, so ``/insights/triggers`` reflects the real prior at zero data.

    Idempotent: components that already have a row are skipped, so re-calling with the
    same conditions creates 0 new rows.

    Args:
        db: Async database session
        user_id: UUID of the onboarding user
        condition_types: List of condition strings (e.g. ["ibs", "mcas"])

    Returns:
        List of newly created TriggerPrediction rows.
    """
    from app.services.hierarchical_trigger import cold_start_results

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

    # Cold-start prior results for only the not-yet-seeded conditions. Pass the
    # matching condition strings so their implicated components carry the clinical seed.
    results = [
        r
        for r in await cold_start_results(db, condition_types)
        if r.component_type in to_seed
    ]
    created = await update_trigger_predictions(db, user_id, results)
    logger.info(
        f"seed_condition_priors: {len(created)} prior rows created for user {user_id}"
    )
    return created
