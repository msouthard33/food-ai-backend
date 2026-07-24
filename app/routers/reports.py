"""Report generation endpoints."""

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.report import ReportGenerateRequest, ReportOut
from app.services import clinician_report_service, report_service

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


@router.get(
    "/clinician-pdf",
    summary="Generate and download a clinician-facing PDF summary (Box 11)",
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}}},
)
async def clinician_pdf(
    lookback_days: int = Query(
        30, ge=7, le=365, description="Days of history to summarize in the report"
    ),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Return a clinician-facing PDF: symptom timeline, the suspect-foods leaderboard
    with confidence + sample size, elimination-protocol outcomes, and patient-reported
    outcomes. Layout follows a documented GI symptom-history intake structure; a real
    GI/allergist review of the layout remains a pending human gate (see W2-5 report)."""
    pdf_bytes = await clinician_report_service.generate_clinician_pdf(
        db, user, lookback_days=lookback_days
    )
    filename = clinician_report_service.clinician_pdf_filename(user.id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
