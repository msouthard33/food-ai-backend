"""Symptom tracking endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import PaginationParams, get_current_user
from app.models.user import User
from app.schemas.symptom import SymptomListOut, SymptomScoreCreate, SymptomScoreOut
from app.services import symptom_service

router = APIRouter(prefix="/api/v1/symptoms", tags=["symptoms"])


@router.post(
    "",
    response_model=SymptomScoreOut,
    status_code=status.HTTP_201_CREATED,
    summary="Log a symptom score",
)
async def create_symptom(
    data: SymptomScoreCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SymptomScoreOut:
    score = await symptom_service.create_symptom_score(db, user.id, data)
    return SymptomScoreOut.model_validate(score)


@router.get(
    "",
    response_model=SymptomListOut,
    summary="List symptoms with date range filter",
)
async def list_symptoms(
    pagination: PaginationParams = Depends(),
    start_date: datetime | None = Query(None, description="Filter from this date (ISO 8601)"),
    end_date: datetime | None = Query(None, description="Filter until this date (ISO 8601)"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SymptomListOut:
    scores, total = await symptom_service.list_symptom_scores(
        db,
        user.id,
        offset=pagination.offset,
        limit=pagination.page_size,
        start_date=start_date,
        end_date=end_date,
    )
    return SymptomListOut(
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        items=[SymptomScoreOut.model_validate(s) for s in scores],
    )
