"""Trigger insights endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
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
