"""User account operations, including full account + PHI erasure (GDPR Art. 17 / CCPA)."""

import logging
import uuid
from urllib.parse import unquote

import httpx
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.meal import Meal
from app.models.user import User

logger = logging.getLogger(__name__)


def _storage_path_from_url(photo_url: str, bucket: str) -> str | None:
    """Best-effort: extract the in-bucket object path from a stored ``photo_url``.

    Handles both a full public/signed Supabase URL (``.../<bucket>/<path>``) and a
    bare object path. Returns ``None`` if no plausible path can be derived.
    """
    if not photo_url:
        return None
    marker = f"/{bucket}/"
    idx = photo_url.find(marker)
    if idx != -1:
        return unquote(photo_url[idx + len(marker):]).split("?", 1)[0].lstrip("/") or None
    if "://" not in photo_url:  # already an object path
        return photo_url.lstrip("/") or None
    return None


async def _delete_meal_photos(paths: list[str]) -> None:
    """Best-effort removal of meal photos from Supabase Storage. Never raises.

    Structured-PHI erasure (the DB rows) must not be blocked by a transient storage
    error, so failures here are logged for manual cleanup rather than propagated.
    """
    settings = get_settings()
    if not paths or not settings.supabase_url or not settings.supabase_service_role_key:
        return
    url = (
        f"{settings.supabase_url.rstrip('/')}/storage/v1/object/"
        f"{settings.supabase_storage_bucket}"
    )
    headers = {
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "apikey": settings.supabase_service_role_key,
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.request("DELETE", url, headers=headers, json={"prefixes": paths})
        if resp.status_code >= 400:
            logger.error(
                "Meal-photo storage cleanup failed (status=%s) for %d object(s); "
                "manual cleanup required",
                resp.status_code, len(paths),
            )
    except httpx.HTTPError as exc:
        logger.error(
            "Meal-photo storage cleanup errored for %d object(s); manual cleanup "
            "required: %s", len(paths), type(exc).__name__,
        )


async def _delete_supabase_auth_user(user_id: uuid.UUID) -> None:
    """Delete the Supabase Auth identity so the account cannot be re-authenticated.

    ``get_current_user`` auto-provisions an app-DB row from the JWT ``sub`` on first
    call, so without removing the Auth identity the person could sign in again and
    resurrect an (empty) account. Best-effort: logged, not raised — the caller has
    already erased the data, and a surviving Auth login is a lesser failure than
    leaving PHI behind.
    """
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return  # dev/testing without Supabase configured
    url = f"{settings.supabase_url.rstrip('/')}/auth/v1/admin/users/{user_id}"
    headers = {
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "apikey": settings.supabase_service_role_key,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.delete(url, headers=headers)
        if resp.status_code >= 400 and resp.status_code != 404:
            logger.error(
                "Supabase Auth user deletion failed (status=%s); the identity can "
                "still sign in — manual removal required",
                resp.status_code,
            )
    except httpx.HTTPError as exc:
        logger.error(
            "Supabase Auth user deletion errored; manual removal required: %s",
            type(exc).__name__,
        )


async def delete_user_account(db: AsyncSession, user: User) -> None:
    """Permanently and irreversibly delete a user and ALL of their data.

    Three cleanup surfaces, in order:
      1. Meal photos in Supabase Storage (the DB cascade cannot reach the bucket).
      2. The Supabase Auth identity (prevents re-auth / auto-reprovisioning).
      3. The ``users`` DB row — a single DELETE cascades via ON DELETE CASCADE to
         every PHI table (meals, meal_items, symptom_scores, daily_checkins,
         medication_logs, trigger_predictions, correlation_events, insight_reports,
         report_shares, user_conditions, user_known_allergens, user_settings,
         user_sensitivity_profiles, food_combined_ratings, ai_conversations).

    Storage and Auth cleanup are best-effort (logged on failure); the DB deletion is
    authoritative and is committed last so a failure there rolls back cleanly.
    """
    settings = get_settings()
    user_id = user.id
    bucket = settings.supabase_storage_bucket

    # Collect photo object paths BEFORE the meal rows are deleted.
    photo_urls = (
        await db.execute(
            select(Meal.photo_url).where(
                Meal.user_id == user_id, Meal.photo_url.is_not(None)
            )
        )
    ).scalars().all()
    paths = [p for u in photo_urls if (p := _storage_path_from_url(u, bucket))]

    await _delete_meal_photos(paths)
    await _delete_supabase_auth_user(user_id)

    await db.execute(delete(User).where(User.id == user_id))
    await db.commit()

    logger.info(
        "Account erased: user_id=%s (%d meal photo(s) queued for storage removal)",
        user_id, len(paths),
    )
