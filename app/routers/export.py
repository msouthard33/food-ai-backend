"""Full user data export — signed-URL delivery (Pillar 4 / Box 13).

Two-step flow so the heavy payload is fetched over a short-lived, unguessable link
rather than inline on the authenticated request:

1. ``POST /api/v1/export/request`` (authenticated) mints an HMAC-signed token that
   encodes the user id + expiry and returns a download URL.
2. ``GET /api/v1/export/download?token=...`` (unauthenticated — the signed token IS
   the credential) validates the token and streams a structured JSON export of all of
   the user's meals, symptoms, medications, and protocols.

Soft-deleted rows (``deleted_at`` set) are excluded from the export.
"""

import base64
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models.meal import Meal
from app.models.medication import MedicationLog
from app.models.sensitivity import UserSensitivityProfile
from app.models.symptom import SymptomScore
from app.models.user import User

router = APIRouter(prefix="/api/v1/export", tags=["insights"])

EXPORT_TTL_SECONDS = 900  # signed link lives for 15 minutes
EXPORT_VERSION = "1.0"
_TOKEN_PURPOSE = "data-export"


# ── signed-token helpers ─────────────────────────────────────────────────────

def _signing_key() -> bytes:
    """HMAC key for export tokens. Falls back to a dev-only key outside production
    (production startup already requires APP_SECRET_KEY to be set — see config.py)."""
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
            status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed export token"
        ) from exc

    expected = _b64e(hmac.new(_signing_key(), body.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid export token signature"
        )

    try:
        payload = json.loads(_b64d(body))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed export token"
        ) from exc

    if payload.get("purpose") != _TOKEN_PURPOSE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid export token purpose"
        )
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(
            status_code=status.HTTP_410_GONE, detail="Export link has expired"
        )
    return payload


# ── endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/request",
    summary="Request a signed URL to download a full data export",
)
async def request_export(
    request: Request,
    user: User = Depends(get_current_user),
) -> dict:
    """Mint a short-lived signed URL the client can use to download its full export."""
    exp = int(time.time()) + EXPORT_TTL_SECONDS
    token = _sign({"uid": str(user.id), "exp": exp, "purpose": _TOKEN_PURPOSE})
    download_url = f"{request.url_for('download_export')}?token={token}"
    return {
        "download_url": download_url,
        "expires_at": datetime.fromtimestamp(exp, tz=timezone.utc).isoformat(),
        "expires_in_seconds": EXPORT_TTL_SECONDS,
    }


@router.get(
    "/download",
    name="download_export",
    summary="Download a full data export via a signed token",
)
async def download_export(
    token: str,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Validate the signed token and return the user's full data export as JSON."""
    payload = _verify(token)
    user_id = uuid.UUID(payload["uid"])

    user = (
        await db.execute(select(User).where(User.id == user_id))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Meals (with items) — non-deleted
    meals = list(
        (
            await db.execute(
                select(Meal)
                .where(Meal.user_id == user_id, Meal.deleted_at.is_(None))
                .options(selectinload(Meal.items))
                .order_by(Meal.timestamp)
            )
        )
        .scalars()
        .unique()
        .all()
    )

    # Symptoms — non-deleted
    symptoms = list(
        (
            await db.execute(
                select(SymptomScore)
                .where(SymptomScore.user_id == user_id, SymptomScore.deleted_at.is_(None))
                .order_by(SymptomScore.timestamp)
            )
        )
        .scalars()
        .all()
    )

    # Medications — non-deleted
    medications = list(
        (
            await db.execute(
                select(MedicationLog)
                .where(MedicationLog.user_id == user_id, MedicationLog.deleted_at.is_(None))
                .order_by(MedicationLog.taken_at)
            )
        )
        .scalars()
        .all()
    )

    # Protocols / sensitivity profiles — non-deleted
    protocols = list(
        (
            await db.execute(
                select(UserSensitivityProfile)
                .where(
                    UserSensitivityProfile.user_id == user_id,
                    UserSensitivityProfile.deleted_at.is_(None),
                )
                .order_by(UserSensitivityProfile.created_at)
            )
        )
        .scalars()
        .all()
    )

    export = {
        "export_version": EXPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user": {
            "id": str(user.id),
            "email": user.email,
            "display_name": user.display_name,
            "timezone": user.timezone,
        },
        "meals": [_serialize_meal(m) for m in meals],
        "symptoms": [_serialize_symptom(s) for s in symptoms],
        "medications": [_serialize_medication(m) for m in medications],
        "protocols": [_serialize_protocol(p) for p in protocols],
        "counts": {
            "meals": len(meals),
            "symptoms": len(symptoms),
            "medications": len(medications),
            "protocols": len(protocols),
        },
    }

    return JSONResponse(
        content=export,
        headers={"Content-Disposition": "attachment; filename=foodai-export.json"},
    )


# ── serializers ──────────────────────────────────────────────────────────────

def _enum_value(value: object) -> str:
    """Return an enum's ``.value`` (or the raw value stringified) for JSON export."""
    return str(getattr(value, "value", value))


def _serialize_meal(meal: Meal) -> dict:
    return {
        "id": str(meal.id),
        "timestamp": meal.timestamp.isoformat(),
        "meal_type": _enum_value(meal.meal_type),
        "raw_description": meal.raw_description,
        "ai_parsed_description": meal.ai_parsed_description,
        "photo_url": meal.photo_url,
        "items": [
            {
                "id": str(item.id),
                "name": item.name,
                "quantity": float(item.quantity) if item.quantity is not None else None,
                "unit": item.unit,
                "preparation_method": item.preparation_method,
            }
            for item in meal.items
        ],
        "created_at": meal.created_at.isoformat() if meal.created_at else None,
    }


def _serialize_symptom(symptom: SymptomScore) -> dict:
    return {
        "id": str(symptom.id),
        "timestamp": symptom.timestamp.isoformat(),
        "symptom_type": _enum_value(symptom.symptom_type),
        "vas_score": int(symptom.vas_score),
        "notes": symptom.notes,
        "meal_id": str(symptom.meal_id) if symptom.meal_id else None,
    }


def _serialize_medication(med: MedicationLog) -> dict:
    return {
        "id": str(med.id),
        "symptom_log_id": str(med.symptom_log_id),
        "medication_name": med.medication_name,
        "dose_mg": float(med.dose_mg) if med.dose_mg is not None else None,
        "taken_at": med.taken_at.isoformat(),
    }


def _serialize_protocol(profile: UserSensitivityProfile) -> dict:
    return {
        "id": str(profile.id),
        "component_type": _enum_value(profile.component_type),
        "weight": float(profile.weight),
        "threshold": float(profile.threshold),
        "active": profile.active,
        "notes": profile.notes,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
    }
