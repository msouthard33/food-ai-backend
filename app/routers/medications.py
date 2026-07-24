"""Medication co-logging endpoints — MCAS differentiator."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.medication import MedicationLog
from app.models.symptom import SymptomScore
from app.models.user import User
from app.schemas.insights import MedicationLogOut, MedicationLogRequest

router = APIRouter(prefix="/api/v1/symptoms", tags=["symptoms"])


@router.post(
    "/medications",
    response_model=MedicationLogOut,
    status_code=status.HTTP_201_CREATED,
    summary="Co-log medication alongside a symptom entry",
)
async def log_medication(
    data: MedicationLogRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MedicationLogOut:
    """Record an antihistamine or other medication alongside a symptom log.

    This is the MCAS-specific differentiator: correlating medication timing
    with symptom severity to surface medication response patterns.
    """
    # Verify the symptom log exists and belongs to this user
    result = await db.execute(
        select(SymptomScore).where(
            SymptomScore.id == data.symptom_log_id,
            SymptomScore.user_id == user.id,
        )
    )
    symptom = result.scalar_one_or_none()
    if not symptom:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Symptom log not found",
        )

    med_log = MedicationLog(
        user_id=user.id,
        symptom_log_id=data.symptom_log_id,
        medication_name=data.medication_name,
        dose_mg=data.dose_mg,
        taken_at=data.taken_at,
    )
    db.add(med_log)
    await db.flush()

    return MedicationLogOut.model_validate(med_log)
