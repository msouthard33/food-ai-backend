"""Meal CRUD operations."""

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.meal import Meal, MealItem, MealItemComponent
from app.models.enums import ComponentSource
from app.models.food import FoodComponentDetail
from app.schemas.meal import MealCreate, MealItemCreate


async def create_meal(db: AsyncSession, user_id: uuid.UUID, data: MealCreate) -> Meal:
    meal = Meal(
        user_id=user_id,
        timestamp=data.timestamp,
        meal_type=data.meal_type,
        raw_description=data.raw_description,
    )
    db.add(meal)
    await db.flush()
    return meal


async def list_meals(
    db: AsyncSession,
    user_id: uuid.UUID,
    offset: int = 0,
    limit: int = 20,
) -> tuple[list[Meal], int]:
    # Count
    count_q = select(func.count()).select_from(Meal).where(Meal.user_id == user_id)
    total = (await db.execute(count_q)).scalar_one()

    # Fetch with eager-loaded items + components
    q = (
        select(Meal)
        .where(Meal.user_id == user_id)
        .options(selectinload(Meal.items).selectinload(MealItem.components))
        .order_by(Meal.timestamp.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(q)
    meals = list(result.scalars().unique().all())
    return meals, total


async def get_meal(db: AsyncSession, meal_id: uuid.UUID, user_id: uuid.UUID) -> Meal | None:
    q = (
        select(Meal)
        .where(Meal.id == meal_id, Meal.user_id == user_id)
        .options(selectinload(Meal.items).selectinload(MealItem.components))
    )
    result = await db.execute(q)
    return result.scalar_one_or_none()


async def add_meal_items(
    db: AsyncSession,
    meal_id: uuid.UUID,
    items_data: list[MealItemCreate],
) -> list[MealItem]:
    created_items: list[MealItem] = []
    for item_data in items_data:
        meal_item = MealItem(
            meal_id=meal_id,
            food_name=item_data.food_name,
            food_database_id=item_data.food_database_id,
            quantity=item_data.quantity,
            unit=item_data.unit,
            preparation_method=item_data.preparation_method,
            brand=item_data.brand,
            confidence_score=item_data.confidence_score,
        )
        db.add(meal_item)
        await db.flush()

        # Auto-populate allergen components from the food database
        if item_data.food_database_id:
            component_q = select(FoodComponentDetail).where(
                FoodComponentDetail.food_id == item_data.food_database_id
            )
            components = (await db.execute(component_q)).scalars().all()
            for comp in components:
                mic = MealItemComponent(
                    meal_item_id=meal_item.id,
                    component_type=comp.component_type,
                    level=comp.level_score or 0,
                    source=ComponentSource.DATABASE,
                )
                db.add(mic)

        created_items.append(meal_item)

    await db.flush()
    return created_items
