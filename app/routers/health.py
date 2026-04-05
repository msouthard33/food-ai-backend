"""Health check and readiness probe endpoints."""

import logging
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.database import async_session_factory

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def health_check() -> dict:
    """Lightweight liveness probe — no dependency checks."""
    return {"status": "ok", "service": "food-ai-backend", "version": "0.1.0"}


@router.get("/ready", summary="Readiness probe")
async def readiness_check() -> JSONResponse:
    """Readiness probe — verifies database and (optionally) Redis connectivity."""
    details: dict[str, str] = {}
    healthy = True

    # --- Database check ---
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        details["database"] = "ok"
    except Exception as exc:
        logger.error("Readiness: database check failed — %s", exc)
        details["database"] = f"error: {exc}"
        healthy = False

    # --- Redis check (only when REDIS_URL is configured) ---
    redis_url = os.getenv("REDIS_URL", "")
    if redis_url:
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(redis_url, socket_connect_timeout=2)
            await r.ping()
            await r.aclose()
            details["redis"] = "ok"
        except Exception as exc:
            logger.error("Readiness: redis check failed — %s", exc)
            details["redis"] = f"error: {exc}"
            healthy = False
    else:
        details["redis"] = "not_configured"

    status_code = 200 if healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if healthy else "degraded", **details},
    )
