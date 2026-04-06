"""Admin endpoints for backend operations and ingestion."""

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.services.food_ingestion import ingest_allergen_knowledge_base

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


def verify_admin_key(x_admin_key: str = Header(..., description="Admin API key")) -> None:
    """Dependency: reject requests that don't carry the correct admin API key."""
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API key not configured on server",
        )
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin API key",
        )


@router.post(
    "/ingest-foods",
    response_model=dict,
    summary="Ingest allergen knowledge base",
    dependencies=[Depends(verify_admin_key)],
)
async def ingest_foods(
    json_path: str | None = Query(None, description="Path to allergen JSON file"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Ingest the allergen knowledge base JSON into the database.

    If json_path is not provided, uses the default path (or Docker fallback).
    Returns the number of foods ingested.

    Requires X-Admin-Key header matching the configured admin_api_key.
    """
    try:
        count = await ingest_allergen_knowledge_base(db, json_path=json_path)
        return {"ingested": count}
    except FileNotFoundError as e:
        logger.error("Allergen knowledge base file not found: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Allergen knowledge base file not found: {e}",
        )
    except Exception as e:
        logger.error("Error ingesting allergen knowledge base: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during food ingestion",
        )
