"""Meal CRUD operations."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.meal import Meal, MealItem, MealItemComponent
from app.models.food import FoodComponentDetail
from app.models.enums import MealType
from app.schemas.meal import MealCreate, MealItemCreate


async def create_meal(db: AsyncSession, user_id: uuid.UUID, data: MealCreate) -> Meal:
    meal = Meal(
        user_id=user_id,
        # Default to current time if not provided; Meal.timestamp is non-nullable
        timestamp=data.timestamp or datetime.now(timezone.utc),
        # Default to SNACK if not provided; Meal.meal_type is non-nullable
        meal_type=data.meal_type or MealType.SNACK,
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
            name=item_data.name,
            food_entry_id=None,  # No food_database_id in MealItemCreate; can be linked later
            quantity=item_data.quantity,
            unit=item_data.unit,
            preparation_method=item_data.preparation_method,
            raw_text=item_data.raw_text,
            # confidence_score uses model default (Decimal("0.0"))
        )
        db.add(meal_item)
        await db.flush()

        # Auto-populate allergen components from the food database if food_entry_id is set
        if meal_item.food_entry_id:
            component_q = select(FoodComponentDetail).where(
                FoodComponentDetail.food_entry_id == meal_item.food_entry_id
            )
            components = (await db.execute(component_q)).scalars().all()
            for comp in components:
                mic = MealItemComponent(
                    meal_item_id=meal_item.id,
                    component_type=comp.component_type,
                    estimated_level=comp.level or 0,
                    # confidence_score uses model default
                )
                db.add(mic)

        created_items.append(meal_item)

    await db.flush()
    return created_items
