"""Trigger prediction engine - detects food sensitivities by correlating meals with symptoms."""

import logging
import statistics
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.enums import ComponentType, TriggerStatus
from app.models.meal import Meal, MealItem, MealItemComponent
from app.models.symptom import SymptomScore
from app.models.trigger import CorrelationEvent, TriggerPrediction

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


async def analyze_correlations(
    db: AsyncSession,
    user_id: uuid.UUID,
    lookback_days: int = 30,
    time_window_before_hours: tuple[float, float] = (0.5, 72),
) -> tuple[dict[ComponentType, list[dict]], dict[ComponentType, int]]:
    """
    Main entry point for trigger analysis.

    Finds all symptoms in the lookback window, correlates them with meals eaten
    in the time window before each symptom, and builds a frequency matrix. It also
    counts the total number of times each component was eaten across the lookback
    window (the base-rate denominator) so downstream scoring can distinguish a food
    that is *usually* followed by a symptom from one that is merely eaten often.

    Args:
        db: Async database session
        user_id: User ID to analyze
        lookback_days: How many days back to look for data
        time_window_before_hours: (min_hours, max_hours) before symptom to search for meals

    Returns:
        (correlations, total_exposures) where:
          - correlations: component_type -> list of pre-symptom correlation events
          - total_exposures: component_type -> total # of meals containing it in the
            lookback window (>= the # of pre-symptom meals; the base-rate denominator)
    """
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Fetch all symptoms in the lookback window
    symptom_query = (
        select(SymptomScore)
        .where(
            and_(
                SymptomScore.user_id == user_id,
                SymptomScore.timestamp >= cutoff_time,
            )
        )
        .order_by(SymptomScore.timestamp)
    )
    symptoms_result = await db.execute(symptom_query)
    symptoms = list(symptoms_result.scalars().all())

    if not symptoms:
        logger.info(f"No symptoms found for user {user_id} in the past {lookback_days} days")
        return {}, {}

    logger.info(f"Found {len(symptoms)} symptoms for user {user_id}")

    # Build correlation matrix: component_type -> list of (symptom_type, severity, time_lag, exposure_level)
    correlations: dict[ComponentType, list[dict]] = {}

    min_hours, max_hours = time_window_before_hours

    for symptom in symptoms:
        # Find meals eaten within the time window before this symptom
        symptom_time = symptom.timestamp
        earliest_meal_time = symptom_time - timedelta(hours=max_hours)
        latest_meal_time = symptom_time - timedelta(hours=min_hours)

        meal_query = (
            select(Meal)
            .where(
                and_(
                    Meal.user_id == user_id,
                    Meal.timestamp >= earliest_meal_time,
                    Meal.timestamp <= latest_meal_time,
                )
            )
            .options(selectinload(Meal.items).selectinload(MealItem.components))
        )
        meals_result = await db.execute(meal_query)
        meals = list(meals_result.scalars().unique().all())

        if not meals:
            logger.debug(f"No meals found in time window for symptom {symptom.id}")
            continue

        # For each meal, extract components and correlate with the symptom
        for meal in meals:
            time_lag_hours = (symptom_time - meal.timestamp).total_seconds() / 3600

            for meal_item in meal.items:
                for component in meal_item.components:
                    component_type = component.component_type
                    exposure_level = (
                        float(component.estimated_level) if component.estimated_level else 0
                    )

                    if component_type not in correlations:
                        correlations[component_type] = []

                    correlations[component_type].append({
                        "symptom_type": symptom.symptom_type,
                        "severity": int(symptom.vas_score),
                        "time_lag_hours": time_lag_hours,
                        "exposure_level": exposure_level,
                        "meal_id": meal.id,
                        "symptom_id": symptom.id,
                        # Absolute symptom time so recency can be judged against "now"
                        # rather than the (correlation-window-capped) time_lag.
                        "symptom_time": symptom_time,
                    })

    # Base-rate denominator: how many times was each component eaten in the lookback
    # window overall (not just before a symptom)? Counted as distinct meals per
    # component so it lines up with the distinct-meal numerator used in scoring.
    all_meals_query = (
        select(Meal)
        .where(
            and_(
                Meal.user_id == user_id,
                Meal.timestamp >= cutoff_time,
            )
        )
        .options(selectinload(Meal.items).selectinload(MealItem.components))
    )
    all_meals_result = await db.execute(all_meals_query)
    all_meals = list(all_meals_result.scalars().unique().all())

    total_exposures: dict[ComponentType, int] = {}
    for meal in all_meals:
        components_in_meal = {
            component.component_type
            for meal_item in meal.items
            for component in meal_item.components
        }
        for component_type in components_in_meal:
            total_exposures[component_type] = total_exposures.get(component_type, 0) + 1

    logger.info(f"Built correlation matrix with {len(correlations)} components")
    return correlations, total_exposures


def _compute_daily_loads(
    meals: list,
    high_load_threshold: float = 60.0,
) -> tuple[dict, dict]:
    """Pure helper: aggregate component loads by calendar day and identify high-load days.

    Extracted from calculate_daily_load_correlations() to allow unit testing without
    a database session.

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


async def calculate_daily_load_correlations(
    db: AsyncSession,
    user_id: uuid.UUID,
    lookback_days: int = 30,
    high_load_threshold: float = 60.0,
    symptom_window_hours: int = 48,
) -> dict[ComponentType, list[dict]]:
    """
    Detect cumulative-dose triggers by aggregating component load across full calendar days.

    Unlike analyze_correlations() which scores per-meal, this function:
    1. Sums estimated_level for each ComponentType across all meals in a calendar day
    2. Identifies days where the daily sum exceeds high_load_threshold
    3. Checks for symptom events in the following symptom_window_hours
    4. Returns correlation data structured identically to analyze_correlations() output
       so it can be passed to calculate_confidence() unchanged.

    Best for: histamine intolerance (bucket theory), MCAS, oxalate/salicylate load.
    Complements (does not replace) per-meal correlations.

    Args:
        db: Async database session
        user_id: User ID to analyse
        lookback_days: How many days back to look for data
        high_load_threshold: Daily component sum >= this value (0-100 scale) triggers a
            high-load day marker. Default 60 = high load for most components.
        symptom_window_hours: How many hours after midnight of the high-load day to search
            for a symptom. Default 48 h captures next-day delayed reactions.

    Returns:
        dict[ComponentType, list[dict]] — same shape as analyze_correlations() correlations
        output; each event dict carries correlation_type="daily_load" and meal_id=None.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Step A — Fetch all meals with components in the lookback window
    meal_query = (
        select(Meal)
        .where(
            and_(
                Meal.user_id == user_id,
                Meal.timestamp >= cutoff,
            )
        )
        .options(selectinload(Meal.items).selectinload(MealItem.components))
    )
    meals_result = await db.execute(meal_query)
    meals = list(meals_result.scalars().unique().all())

    if not meals:
        logger.info(f"calculate_daily_load_correlations: no meals for user {user_id}")
        return {}

    # Steps B + C — Aggregate by calendar day and find high-load days (pure, testable)
    daily_loads, high_load_days = _compute_daily_loads(meals, high_load_threshold)

    if not high_load_days:
        logger.info(
            f"calculate_daily_load_correlations: no high-load days (threshold={high_load_threshold}) "
            f"for user {user_id}"
        )
        return {}

    # Step D — Fetch all symptoms in the lookback window
    symptom_query = (
        select(SymptomScore)
        .where(
            and_(
                SymptomScore.user_id == user_id,
                SymptomScore.timestamp >= cutoff,
            )
        )
        .order_by(SymptomScore.timestamp)
    )
    symptoms_result = await db.execute(symptom_query)
    symptoms = list(symptoms_result.scalars().all())

    if not symptoms:
        return {}

    # Build correlation events: for each high-load day, find symptoms in the window after
    daily_load_correlations: dict[ComponentType, list[dict]] = {}

    for comp_type, days in high_load_days.items():
        for day in days:
            day_midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            window_end = day_midnight + timedelta(hours=symptom_window_hours)
            daily_sum = daily_loads[day][comp_type]

            for symptom in symptoms:
                if day_midnight <= symptom.timestamp <= window_end:
                    time_lag_hours = (symptom.timestamp - day_midnight).total_seconds() / 3600

                    if comp_type not in daily_load_correlations:
                        daily_load_correlations[comp_type] = []

                    daily_load_correlations[comp_type].append({
                        "symptom_type": symptom.symptom_type,
                        "severity": int(symptom.vas_score),
                        "time_lag_hours": time_lag_hours,
                        "exposure_level": daily_sum,
                        "meal_id": None,  # no single meal to attribute
                        "symptom_id": symptom.id,
                        "symptom_time": symptom.timestamp,
                        "correlation_type": "daily_load",
                    })

    logger.info(
        f"calculate_daily_load_correlations: {len(daily_load_correlations)} components "
        f"with high-load day correlations for user {user_id}"
    )
    return daily_load_correlations


async def calculate_confidence(
    correlations: dict[ComponentType, list[dict]],
    total_exposures: dict[ComponentType, int] | None = None,
) -> dict[ComponentType, float]:
    """
    Score each component based on correlation frequency, severity, consistency, and recency.

    Confidence formula (total 0-100):
    - frequency_score: base-rate corrected -> (meals_before_symptom / total_times_eaten) * 40.
      Answers "of all the times this component was eaten, how often was it followed by a
      symptom?" rather than rewarding foods that are simply eaten a lot.
    - severity_score: (avg_severity_when_present / 100) * 25
    - consistency_score: 25 * (1 - min(stdev(severity) / 30, 1)). A genuine trigger produces
      *consistent* symptom severity (low spread); erratic severity scores lower. Needs >=2
      events to establish consistency, otherwise 0.
    - recency_score: 10 if the component has at least one supporting event whose symptom
      occurred in the last 14 days (judged against absolute time, not the capped time_lag).

    Args:
        correlations: component_type -> list of pre-symptom correlation events
        total_exposures: component_type -> total # of meals containing it in the lookback
            window (base-rate denominator). Falls back to the pre-symptom meal count when
            absent, which collapses the base-rate term to 1.0 (legacy behavior).
    """
    confidence_scores: dict[ComponentType, float] = {}
    total_exposures = total_exposures or {}

    now = datetime.now(timezone.utc)
    recency_cutoff = now - timedelta(days=14)

    for component_type, events in correlations.items():
        if not events:
            continue

        # Frequency score (base-rate corrected): fraction of this component's total
        # exposures that were followed by a symptom. Numerator and denominator both
        # count DISTINCT events so a meal/day followed by several symptoms isn't
        # over-counted. For per-meal events meal_id is used as the distinct key; for
        # daily-load events (meal_id=None) symptom_id is used instead, since each
        # daily-load event represents a distinct high-load day → symptom pair.
        distinct_exposure_keys = {
            e["meal_id"] if e.get("meal_id") is not None else e["symptom_id"]
            for e in events
        }
        meals_before_symptom = len(distinct_exposure_keys)
        total_times_eaten = max(total_exposures.get(component_type, meals_before_symptom), 1)
        base_rate = min(meals_before_symptom / total_times_eaten, 1.0)
        frequency_score = base_rate * 40

        # Severity score: average symptom severity when component was present
        avg_severity = sum(e["severity"] for e in events) / len(events)
        severity_score = (avg_severity / 100) * 25

        # Consistency score: reward low variance in symptom severity. A real trigger
        # tends to produce a repeatable reaction; a wide spread suggests coincidence.
        severities = [e["severity"] for e in events]
        if len(severities) >= 2:
            severity_stdev = statistics.pstdev(severities)
            consistency_score = 25 * (1 - min(severity_stdev / 30.0, 1.0))
        else:
            consistency_score = 0.0

        # Recency score: boost if there is fresh (<=14 day) supporting evidence.
        has_recent_evidence = any(e.get("symptom_time", now) >= recency_cutoff for e in events)
        recency_score = 10 if has_recent_evidence else 0

        total_score = frequency_score + severity_score + consistency_score + recency_score
        confidence_scores[component_type] = min(100, max(0, total_score))

        logger.debug(
            f"Component {component_type}: freq={frequency_score:.1f} "
            f"(base_rate={meals_before_symptom}/{total_times_eaten}), "
            f"sev={severity_score:.1f}, cons={consistency_score:.1f}, "
            f"recency={recency_score:.1f} => confidence={total_score:.1f}"
        )

    # NOTE: When incorporating population/synthetic data in future,
    # weight real observations at 1.0 and synthetic at 0.1 (10x real-observation weight).
    # This prevents the population prior from swamping a user's personal signal.
    # See synthetic_data_generator.py for implementation pattern.
    return confidence_scores


async def update_trigger_predictions(
    db: AsyncSession,
    user_id: uuid.UUID,
    confidence_scores: dict[ComponentType, float],
    correlation_data: dict[ComponentType, list[dict]],
) -> list[TriggerPrediction]:
    """
    Upsert trigger predictions based on confidence scores.

    For each component with confidence > 20:
    - Create or update TriggerPrediction
    - Set status: suspect (20-49), probable (50-74), confirmed (75+)
    - Components that drop below 15 get status "cleared"
    - Create CorrelationEvent records for each meal-symptom pair
    - Track symptom_types array and average_time_lag_minutes
    """
    updated_predictions: list[TriggerPrediction] = []

    # Process components with meaningful confidence scores
    for component_type, confidence in confidence_scores.items():
        if confidence < 15:
            # Mark as cleared if it was previously tracked
            existing = await db.execute(
                select(TriggerPrediction).where(
                    and_(
                        TriggerPrediction.user_id == user_id,
                        TriggerPrediction.component_type == component_type,
                    )
                )
            )
            existing_pred = existing.scalar_one_or_none()
            if existing_pred:
                existing_pred.status = TriggerStatus.CLEARED
                existing_pred.last_updated = datetime.now(timezone.utc)
            continue

        if confidence < 20:
            continue

        # Determine status based on confidence
        if confidence >= 75:
            status = TriggerStatus.CONFIRMED
        elif confidence >= 50:
            status = TriggerStatus.PROBABLE
        else:
            status = TriggerStatus.SUSPECT

        # Get or create trigger prediction
        existing = await db.execute(
            select(TriggerPrediction).where(
                and_(
                    TriggerPrediction.user_id == user_id,
                    TriggerPrediction.component_type == component_type,
                )
            )
        )
        prediction = existing.scalar_one_or_none()

        if not prediction:
            prediction = TriggerPrediction(
                user_id=user_id,
                component_type=component_type,
                confidence_score=Decimal(str(round(confidence, 1))),
                status=status,
                first_detected=datetime.now(timezone.utc),
            )
            db.add(prediction)
            await db.flush()
            logger.info(f"Created new trigger prediction for {component_type} with confidence {confidence:.1f}")
        else:
            prediction.confidence_score = Decimal(str(round(confidence, 1)))
            prediction.status = status
            prediction.last_updated = datetime.now(timezone.utc)

        # Track symptom types and time lags
        events = correlation_data.get(component_type, [])
        if events:
            # Extract unique symptom types
            symptom_types_set = {str(e["symptom_type"]) for e in events}
            prediction.symptom_types = sorted(list(symptom_types_set))

            # Calculate average time lag in minutes
            avg_time_lag_hours = sum(e["time_lag_hours"] for e in events) / len(events)
            prediction.average_time_lag_minutes = int(avg_time_lag_hours * 60)

            prediction.evidence_count = len(events)

        await db.flush()
        updated_predictions.append(prediction)

        # Create or update CorrelationEvent records.
        # Daily-load events (meal_id=None) cannot produce a CorrelationEvent row because
        # the FK constraint on correlation_events.meal_id is NOT NULL. They still
        # contribute to confidence scoring and symptom_types/time_lag accounting above.
        for event_data in events:
            meal_id = event_data["meal_id"]
            if meal_id is None:
                # daily_load correlation — no single meal to attribute; skip row
                continue
            symptom_id = event_data["symptom_id"]
            time_lag = event_data["time_lag_hours"]
            exposure = event_data["exposure_level"]
            severity = event_data["severity"]

            # Check if correlation event already exists
            existing_event = await db.execute(
                select(CorrelationEvent).where(
                    and_(
                        CorrelationEvent.trigger_prediction_id == prediction.id,
                        CorrelationEvent.meal_id == meal_id,
                        CorrelationEvent.symptom_score_id == symptom_id,
                    )
                )
            )
            correlation_event = existing_event.scalar_one_or_none()

            if not correlation_event:
                correlation_event = CorrelationEvent(
                    trigger_prediction_id=prediction.id,
                    meal_id=meal_id,
                    symptom_score_id=symptom_id,
                    time_lag_hours=Decimal(str(round(time_lag, 2))),
                    component_exposure_level=Decimal(str(round(exposure, 1))),
                    symptom_severity=severity,
                )
                db.add(correlation_event)

        await db.flush()

    await db.commit()
    logger.info(f"Updated {len(updated_predictions)} trigger predictions for user {user_id}")
    return updated_predictions


async def run_full_analysis(
    db: AsyncSession,
    user_id: uuid.UUID,
    lookback_days: int = 30,
) -> list[TriggerPrediction]:
    """
    Convenience wrapper that orchestrates the full trigger analysis pipeline.

    1. analyze_correlations: Find all per-meal correlations (existing)
    2. calculate_daily_load_correlations: Find cumulative-dose correlations (new)
    3. Merge both into a unified correlation map
    4. calculate_confidence: Score each component
    5. update_trigger_predictions: Persist results and create correlation events
    """
    logger.info(f"Starting full trigger analysis for user {user_id} (lookback: {lookback_days} days)")

    # Step 1: Per-meal correlations + per-component exposure base rates
    per_meal_correlations, total_exposures = await analyze_correlations(db, user_id, lookback_days)

    # Step 2: Daily-load correlations (cumulative-dose triggers)
    daily_load_correlations = await calculate_daily_load_correlations(db, user_id, lookback_days)

    # Step 3: Merge — for each ComponentType, combine event lists from both sources
    merged_correlations: dict[ComponentType, list[dict]] = {}
    all_component_types = set(per_meal_correlations) | set(daily_load_correlations)
    for ct in all_component_types:
        merged_correlations[ct] = (
            per_meal_correlations.get(ct, []) + daily_load_correlations.get(ct, [])
        )

    if not merged_correlations:
        logger.info(f"No correlations found for user {user_id}")
        return []

    # Step 4: Calculate confidence scores (base-rate corrected; handles both event types)
    confidence_scores = await calculate_confidence(merged_correlations, total_exposures)

    # Step 5: Update predictions in database
    predictions = await update_trigger_predictions(
        db, user_id, confidence_scores, merged_correlations
    )

    logger.info(f"Completed trigger analysis for user {user_id}: {len(predictions)} predictions")
    return predictions


# ── Synthetic data decay rule ─────────────────────────────────────────────────

SYNTHETIC_DECAY_THRESHOLD = 42
"""Number of real (non-synthetic) meal log entries before synthetic population
priors are excluded from trigger inference for a given user.

Design rationale:
    Synthetic patients are separate User rows (is_synthetic=True) inserted by
    generate_synthetic_cohort(). They are NOT co-mingled with real user data;
    instead, the trigger engine can optionally weight population-derived priors
    during cold-start (the real user's first 0–42 meals).

    Once a real user's own meal count exceeds SYNTHETIC_DECAY_THRESHOLD, their
    trigger predictions are fully data-driven and population priors should be
    given zero weight. Enforcement options:
        a) seed_condition_priors() checks get_real_meal_count() and skips
           population prior insertion if threshold is exceeded.
        b) The insights endpoint filters TriggerPrediction rows whose notes
           field contains "source: kb_prior" once the threshold is met.

    42 meals ≈ 14 days of consistent 3-meals-per-day logging — enough signal
    for personal trigger patterns to emerge.
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
    """Seed initial TriggerPrediction rows based on a user's declared conditions.

    Called at onboarding to give the trigger engine a cold-start signal before
    the user has logged enough meals and symptoms for statistical confidence.

    Each condition maps to a set of ComponentTypes via CONDITION_PRIORS. Rows are
    seeded with confidence_score=18 (just below the 20-threshold for active display)
    and status=SUSPECT so they appear in the queue but don't surface prematurely.

    Idempotent: silently skips any (user_id, component_type) pair that already has
    a TriggerPrediction row.

    Args:
        db: Async database session
        user_id: UUID of the onboarding user
        condition_types: List of condition strings (e.g. ["ibs", "mcas"])

    Returns:
        List of newly created TriggerPrediction rows; skipped rows are not included.
    """
    # Collect unique component types across all declared conditions
    components_to_seed: set[ComponentType] = set()
    for condition in condition_types:
        priors = CONDITION_PRIORS.get(condition.lower().strip(), [])
        components_to_seed.update(priors)

    if not components_to_seed:
        logger.info(
            f"seed_condition_priors: no known conditions in {condition_types!r} for user {user_id}"
        )
        return []

    created: list[TriggerPrediction] = []

    for component_type in components_to_seed:
        # Skip if a prediction already exists for this user/component pair
        existing = await db.execute(
            select(TriggerPrediction).where(
                and_(
                    TriggerPrediction.user_id == user_id,
                    TriggerPrediction.component_type == component_type,
                )
            )
        )
        if existing.scalar_one_or_none() is not None:
            logger.debug(
                f"seed_condition_priors: skipping {component_type} — already exists for user {user_id}"
            )
            continue

        prediction = TriggerPrediction(
            user_id=user_id,
            component_type=component_type,
            confidence_score=Decimal("18"),
            status=TriggerStatus.SUSPECT,
            first_detected=datetime.now(timezone.utc),
            evidence_count=0,
        )
        db.add(prediction)
        created.append(prediction)
        logger.info(f"seed_condition_priors: seeded {component_type} for user {user_id}")

    if created:
        await db.flush()
        await db.commit()
        logger.info(f"seed_condition_priors: {len(created)} rows created for user {user_id}")

    return created
