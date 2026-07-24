"""Trigger insights endpoints — lag correlation, suspect foods, triggers."""

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.dependencies import get_current_user
from app.models.meal import Meal, MealItem, MealItemComponent
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

router = APIRouter(prefix="/api/v1/insights", tags=["insights"])


@router.get(
    "/triggers",
    response_model=TriggerListOut,
    summary="Get trigger predictions for the authenticated user",
)
async def get_triggers(
    status_filter: str | None = Query(
        None, alias="status", description="Filter by trigger status (suspect, probable, confirmed, cleared)"
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
    appeared in a meal within the lag window before the symptom event.
    Only tuples with sample_size >= 2 are returned.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    windows = [24, 48, 72]
    rows: list[LagCorrelationRow] = []

    # Fetch all symptoms in range
    symptom_result = await db.execute(
        select(SymptomScore)
        .where(and_(SymptomScore.user_id == user.id, SymptomScore.timestamp >= cutoff))
        .order_by(SymptomScore.timestamp)
    )
    symptoms = list(symptom_result.scalars().all())

    if not symptoms:
        return LagCorrelationOut(correlations=[], total=0)

    # Fetch all meals with items in range (go back further to cover max window)
    meal_cutoff = cutoff - timedelta(hours=max(windows))
    meal_result = await db.execute(
        select(Meal)
        .where(and_(Meal.user_id == user.id, Meal.timestamp >= meal_cutoff))
        .options(selectinload(Meal.items))
    )
    meals = list(meal_result.scalars().unique().all())

    # Build correlation buckets: (window, food_name, symptom_type) -> count
    buckets: dict[tuple[int, str, str], int] = {}

    for symptom in symptoms:
        for window_hours in windows:
            window_start = symptom.timestamp - timedelta(hours=window_hours)
            for meal in meals:
                if window_start <= meal.timestamp <= symptom.timestamp:
                    for item in meal.items:
                        key = (window_hours, item.name, str(symptom.symptom_type))
                        buckets[key] = buckets.get(key, 0) + 1

    for (window_hours, food_name, symptom_name), sample_size in buckets.items():
        if sample_size >= 2:
            # Correlation score: simple frequency normalized to 0-100
            total_symptom_count = len(symptoms)
            score = min(100.0, (sample_size / max(total_symptom_count, 1)) * 100)
            rows.append(
                LagCorrelationRow(
                    window_hours=window_hours,
                    food_name=food_name,
                    symptom_name=symptom_name,
                    correlation_score=round(score, 2),
                    sample_size=sample_size,
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
    """Return foods ranked by trigger correlation score.

    Only foods with sample_size >= 3 are included.
    confidence_tier uses the D9 mapping.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Fetch symptoms
    symptom_result = await db.execute(
        select(SymptomScore)
        .where(and_(SymptomScore.user_id == user.id, SymptomScore.timestamp >= cutoff))
    )
    symptoms = list(symptom_result.scalars().all())

    if not symptoms:
        return SuspectFoodsOut(foods=[], total=0)

    # Fetch meals with items
    meal_result = await db.execute(
        select(Meal)
        .where(and_(Meal.user_id == user.id, Meal.timestamp >= cutoff - timedelta(hours=72)))
        .options(selectinload(Meal.items))
    )
    meals = list(meal_result.scalars().unique().all())

    # Count how many symptom events each food preceded (within 72h window)
    food_counts: dict[str, int] = {}
    for symptom in symptoms:
        window_start = symptom.timestamp - timedelta(hours=72)
        seen_foods: set[str] = set()
        for meal in meals:
            if window_start <= meal.timestamp <= symptom.timestamp:
                for item in meal.items:
                    seen_foods.add(item.name)
        for food_name in seen_foods:
            food_counts[food_name] = food_counts.get(food_name, 0) + 1

    total_symptom_events = len(symptoms)
    result_foods: list[SuspectFoodRow] = []

    for food_name, sample_size in food_counts.items():
        if sample_size < 3:
            continue
        trigger_score = min(100.0, (sample_size / max(total_symptom_events, 1)) * 100)
        result_foods.append(
            SuspectFoodRow(
                food_name=food_name,
                trigger_score=round(trigger_score, 2),
                sample_size=sample_size,
            )
        )

    result_foods.sort(key=lambda f: f.trigger_score, reverse=True)
    return SuspectFoodsOut(foods=result_foods, total=len(result_foods))
