"""Data import endpoints — Pillar 4 / Box 12 (Clinician Trust Layer).

Lets a migrating user bring their history in from a competitor app. Currently
supports the mySymptoms (SkyGazer Labs) CSV export format.
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.services import csv_import_service

router = APIRouter(prefix="/api/v1/import", tags=["import"])

# Guard against oversized uploads (a personal diary export is small).
_MAX_CSV_BYTES = 5 * 1024 * 1024


@router.post(
    "/csv",
    status_code=status.HTTP_200_OK,
    summary="Import a mySymptoms CSV export into meals + symptoms",
)
async def import_csv(
    file: UploadFile = File(..., description="mySymptoms CSV export file"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Parse an uploaded mySymptoms CSV export and create the corresponding meals and
    symptoms for the authenticated user.

    Returns a per-import summary: ``total_rows``, ``meals_created``,
    ``symptoms_created``, ``rows_skipped``, and a list of row-level ``errors``. The
    import is idempotent-friendly — re-uploading the same file will not duplicate
    already-imported rows.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uploaded file is empty"
        )
    if len(raw) > _MAX_CSV_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"CSV exceeds {_MAX_CSV_BYTES // (1024 * 1024)} MB limit",
        )

    result = await csv_import_service.import_mysymptoms_csv(db, user.id, raw)
    return {"source_format": "mysymptoms", **result.as_dict()}
