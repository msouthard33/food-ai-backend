"""Report generation endpoints."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.report import ReportGenerateRequest, ReportOut
from app.services import report_service

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.post(
    "/generate",
    response_model=ReportOut,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a summary report for a date range",
)
async def generate_report(
    data: ReportGenerateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ReportOut:
    report = await report_service.generate_report(db, user.id, data)
    return ReportOut.model_validate(report)
