"""Report generation endpoints."""

import base64
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.report import ReportGenerateRequest, ReportOut
from app.services import (
    clinician_report_service,
    report_service,
    summary_report_service,
)

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])

# ── Signed-share config (mirrors export.py's HMAC signed-URL pattern) ──────────
SUMMARY_SHARE_TTL_SECONDS = 900  # signed share link lives for 15 minutes
_SUMMARY_SHARE_PURPOSE = "summary-share"


def _signing_key() -> bytes:
    """HMAC key for summary-share tokens (same key source as data-export tokens)."""
    key = get_settings().app_secret_key or "insecure-dev-export-key"
    return key.encode("utf-8")


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(encoded: str) -> bytes:
    pad = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded + pad)


def _sign(payload: dict) -> str:
    body = _b64e(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    sig = hmac.new(_signing_key(), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64e(sig)}"


def _verify(token: str) -> dict:
    try:
        body, sig = token.split(".", 1)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed share token"
        ) from exc

    expected = _b64e(hmac.new(_signing_key(), body.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid share token signature"
        )
    try:
        payload = json.loads(_b64d(body))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed share token"
        ) from exc

    if payload.get("purpose") != _SUMMARY_SHARE_PURPOSE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid share token purpose"
        )
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(
            status_code=status.HTTP_410_GONE, detail="Share link has expired"
        )
    return payload


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


@router.get(
    "/summary-pdf",
    summary="Generate and download the shareable doctor/patient summary PDF (Pillar 4)",
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}}},
)
async def summary_pdf(
    lookback_days: int = Query(
        30, ge=7, le=365, description="Days of history to summarize in the report"
    ),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Return the shareable doctor/patient summary PDF: symptom timeline, the
    engine-agnostic food-signal section (from the ``build_summary_signal_rows`` seam),
    elimination-protocol status, and patient-reported outcomes."""
    pdf_bytes = await summary_report_service.generate_summary_pdf(
        db, user, lookback_days=lookback_days
    )
    filename = summary_report_service.summary_pdf_filename(user.id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/summary/share",
    summary="Mint a short-lived signed URL to share the doctor/patient summary PDF",
)
async def request_summary_share(
    request: Request,
    lookback_days: int = Query(
        30, ge=7, le=365, description="Days of history the shared PDF will summarize"
    ),
    user: User = Depends(get_current_user),
) -> dict:
    """Return an HMAC-signed, 15-minute download URL (the token IS the credential),
    following the same signed-URL pattern as the data-export flow."""
    exp = int(time.time()) + SUMMARY_SHARE_TTL_SECONDS
    token = _sign(
        {
            "uid": str(user.id),
            "exp": exp,
            "lookback_days": int(lookback_days),
            "purpose": _SUMMARY_SHARE_PURPOSE,
        }
    )
    share_url = f"{request.url_for('download_shared_summary')}?token={token}"
    return {
        "share_url": share_url,
        "expires_at": datetime.fromtimestamp(exp, tz=timezone.utc).isoformat(),
        "expires_in_seconds": SUMMARY_SHARE_TTL_SECONDS,
    }


@router.get(
    "/summary/shared",
    name="download_shared_summary",
    summary="Download a shared doctor/patient summary PDF via a signed token",
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}}},
)
async def download_shared_summary(
    token: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Validate the signed token (no auth header required — the token is the
    credential) and stream the summary PDF for the encoded user + window."""
    payload = _verify(token)
    user_id = uuid.UUID(payload["uid"])
    lookback_days = int(payload.get("lookback_days", 30))

    user = (
        await db.execute(select(User).where(User.id == user_id))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    pdf_bytes = await summary_report_service.generate_summary_pdf(
        db, user, lookback_days=lookback_days
    )
    filename = summary_report_service.summary_pdf_filename(user_id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
