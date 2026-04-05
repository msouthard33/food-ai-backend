"""Trigger prediction engine - detects food sensitivities by correlating meals with symptoms."""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.enums import ComponentType, TriggerStatus
from app.models.meal import Meal, MealItem, MealItemComponent
from app.models.symptom import SymptomScore
from app.models.trigger import CorrelationEvent, TriggerPrediction

logger = logging.getLogger(__name__)


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
    time_window_before_hours: tuple[float, float] = (0.5, 24),
) -> dict:
    """
    Main entry point for trigger analysis.

    Finds all symptoms in the lookback window, correlates them with meals eaten
    in the time window before each symptom, and builds a frequency matrix.

    Args:
        db: Async database session
        user_id: User ID to analyze
        lookback_days: How many days back to look for data
        time_window_before_hours: (min_hours, max_hours) before symptom to search for meals

    Returns:
        Dict with correlation data indexed by component_type
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
        return {}

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
                    exposure_level = float(component.level) if component.level else 0

                    if component_type not in correlations:
                        correlations[component_type] = []

                    correlations[component_type].append({
                        "symptom_type": symptom.symptom_type,
                        "severity": int(symptom.vas_score),
                        "time_lag_hours": time_lag_hours,
                        "exposure_level": exposure_level,
                        "meal_id": meal.id,
                        "symptom_id": symptom.id,
                    })

    logger.info(f"Built correlation matrix with {len(correlations)} components")
    return correlations


async def calculate_confidence(
    correlations: dict[ComponentType, list[dict]],
    total_symptom_events: int = 1,
) -> dict[ComponentType, float]:
    """
    Score each component based on correlation frequency and severity.

    Confidence formula:
    - frequency_score: (times_associated / total_symptom_events) * 40
    - severity_score: (avg_severity_when_present / 100) * 25
    - consistency_score: (percentage of occurrences with this component) * 25
    - recency_score: boost for correlations in last 7 days vs older ones * 10

    Total confidence 0-100.
    """
    confidence_scores: dict[ComponentType, float] = {}

    if total_symptom_events == 0:
        total_symptom_events = 1

    for component_type, events in correlations.items():
        if not events:
            continue

        # Frequency score: how many times this component appeared before symptoms
        frequency_score = min(100, (len(events) / total_symptom_events) * 40)

        # Severity score: average severity when component was present
        avg_severity = sum(e["severity"] for e in events) / len(events)
        severity_score = (avg_severity / 100) * 25

        # Consistency score: what proportion of events involved this component
        consistency_score = min(25, (len(events) / max(total_symptom_events, 1)) * 25)

        # Recency score: boost for recent correlations
        recent_events = [e for e in events if e.get("time_lag_hours", 0) <= (7 * 24)]
        recency_score = 10 if recent_events else 0

        total_score = frequency_score + severity_score + consistency_score + recency_score
        confidence_scores[component_type] = min(100, max(0, total_score))

        logger.debug(
            f"Component {component_type}: freq={frequency_score:.1f}, "
            f"sev={severity_score:.1f}, cons={consistency_score:.1f}, "
            f"recency={recency_score:.1f} => confidence={total_score:.1f}"
        )

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

        # Create or update CorrelationEvent records
        for event_data in events:
            meal_id = event_data["meal_id"]
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

    1. analyze_correlations: Find all meal-symptom correlations
    2. calculate_confidence: Score each component
    3. update_trigger_predictions: Persist results and create correlation events
    """
    logger.info(f"Starting full trigger analysis for user {user_id} (lookback: {lookback_days} days)")

    # Step 1: Build correlation matrix
    correlations = await analyze_correlations(db, user_id, lookback_days)

    if not correlations:
        logger.info(f"No correlations found for user {user_id}")
        return []

    # Count total symptom events for scoring context
    symptom_query = (
        select(func.count())
        .select_from(SymptomScore)
        .where(
            and_(
                SymptomScore.user_id == user_id,
                SymptomScore.timestamp
                >= datetime.now(timezone.utc) - timedelta(days=lookback_days),
            )
        )
    )
    total_symptoms = (await db.execute(symptom_query)).scalar() or 1

    # Step 2: Calculate confidence scores
    confidence_scores = await calculate_confidence(correlations, total_symptom_events=total_symptoms)

    # Step 3: Update predictions in database
    predictions = await update_trigger_predictions(db, user_id, confidence_scores, correlations)

    logger.info(f"Completed trigger analysis for user {user_id}: {len(predictions)} predictions")
    return predictions
