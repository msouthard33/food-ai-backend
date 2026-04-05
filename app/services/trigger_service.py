"""Trigger prediction queries."""

import uuid

from sqlalchem import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.trigger import TriggerPrediction


async def get_user_triggers(
    db: AsyncSession,
    user_id: uuid.UUID,
    status_filter: str | None = None,
) -> list[TriggerPrediction]:
    q = select(TriggerPrediction).where(TriggerPrediction.user_id == user_id)
    if status_filter:
        q = q.where(TriggerPrediction.status == status_filter)
    q = q.order_by(TriggerPrediction.confidence_score.desc())
    result = await db.execute(q)
    return list(result.scalars().all())
