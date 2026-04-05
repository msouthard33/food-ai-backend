"""Admin endpoints for backend operations and ingestion."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.food_ingestion import ingest_allergen_knowledge_base

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.post(
    "/ingest-foods",
    response_model=dict,
    summary="Ingest allergen knowledge base",
)
async def ingest_foods(
    json_path: str | None = Query(None, description="Path to allergen JSON file"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Ingest the allergen knowledge base JSON into the database.

    If json_path is not provided, uses the default path (or Docker fallback).
    Returns the number of foods ingested.
    """
    try:
        count = await ingest_allergen_knowledge_base(db, json_path=json_path)
        return {"ingested": count}
    except Exception as e:
        logger.error("Error ingesting allergen knowledge base: %s", e)
        raise
