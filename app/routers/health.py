"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="Health check")
async def health_check() -> dict:
    return {"status": "ok", "service": "food-ai-backend", "version": "0.1.0"}
