"""Symptom score CRUD operations."""

import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.symptom import SymptomScore
from app.schemas.symptom import SymptomScoreCreate


async def create_symptom_score(
    db: AsyncSession,
    user_id: uuid.UUID,
    data: SymptomScoreCreate,
) -> SymptomScore:
    score = SymptomScore(
        user_id=user_id,
        meal_id=data.meal_id,
        timestamp=data.timestamp,
        symptom_type=data.symptom_type,
        vas_score=data.vas_score,
        notes=data.notes,
        prompt_type=data.prompt_type,
    )
    db.add(score)
    await db.flush()
    return score


async def list_symptom_scores(
    db: AsyncSession,
    user_id: uuid.UUID,
    offset: int = 0,
    limit: int = 20,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> tuple[list[SymptomScore], int]:
    base = select(SymptomScore).where(SymptomScore.user_id == user_id)
    count_base = select(func.count()).select_from(SymptomScore).where(SymptomScore.user_id == user_id)

    if start_date:
        base = base.where(SymptomScore.timestamp >= start_date)
        count_base = count_base.where(SymptomScore.timestamp >= start_date)
    if end_date:
        base = base.where(SymptomScore.timestamp <= end_date)
        count_base = count_base.where(SymptomScore.timestamp <= end_date)

    total = (await db.execute(count_base)).scalar_one()
    q = base.order_by(SymptomScore.timestamp.desc()).offset(offset).limit(limit)
    result = await db.execute(q)
    scores = list(result.scalars().all())
    return scores, total
