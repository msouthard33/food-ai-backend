"""Food knowledge base search."""

import uuid

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.food import FoodEntry


async def search_foods(
    db: AsyncSession,
    query: str,
    category: str | None = None,
    limit: int = 20,
) -> tuple[list[FoodEntry], int]:
    """Full-text search across food name, common names, category, and subcategory."""
    pattern = f"%{query.lower()}%"

    base = select(FoodEntry).where(
        or_(
            func.lower(FoodEntry.name).like(pattern),
            func.lower(FoodEntry.category).like(pattern),
            func.lower(FoodEntry.subcategory).like(pattern),
            FoodEntry.common_names.any(func.lower(query)),
        )
    )
    count_base = select(func.count()).select_from(FoodEntry).where(
        or_(
            func.lower(FoodEntry.name).like(pattern),
            func.lower(FoodEntry.category).like(pattern),
            func.lower(FoodEntry.subcategory).like(pattern),
            FoodEntry.common_names.any(func.lower(query)),
        )
    )

    if category:
        base = base.where(func.lower(FoodEntry.category) == category.lower())
        count_base = count_base.where(func.lower(FoodEntry.category) == category.lower())

    total = (await db.execute(count_base)).scalar_one()
    # Fix: use FoodEntry.components (correct relationship name), not FoodEntry.component_details
    q = base.options(selectinload(FoodEntry.components)).limit(limit)
    result = await db.execute(q)
    foods = list(result.scalars().unique().all())
    return foods, total


async def get_food_by_id(db: AsyncSession, food_id: uuid.UUID) -> FoodEntry | None:
    q = (
        select(FoodEntry)
        .where(FoodEntry.id == food_id)
        .options(selectinload(FoodEntry.components))
    )
    result = await db.execute(q)
    return result.scalar_one_or_none()
