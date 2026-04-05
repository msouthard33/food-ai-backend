"""Food database search endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.food import FoodSearchListOut, FoodSearchResult
from app.services import food_service

router = APIRouter(prefix="/api/v1/foods", tags=["foods"])


@router.get(
    "/search",
    response_model=FoodSearchListOut,
    summary="Search the food knowledge base",
)
async def search_foods(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    category: str | None = Query(None, description="Filter by food category"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FoodSearchListOut:
    foods, total = await food_service.search_foods(db, q, category=category, limit=limit)
    return FoodSearchListOut(
        total=total,
        query=q,
        items=[FoodSearchResult.model_validate(f) for f in foods],
    )
