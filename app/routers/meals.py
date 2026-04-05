"""Meal logging endpoints."""

import base64
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import PaginationParams, get_current_user
from app.models.enums import ProcessingStatus
from app.models.user import User
from app.schemas.meal import MealCreate, MealItemBatchCreate, MealItemOut, MealListOut, MealOut
from app.services import meal_service

router = APIRouter(prefix="/api/v1/meals", tags=["meals"])


@router.post(
    "",
    response_model=MealOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new meal entry",
)
async def create_meal(
    data: MealCreate,
    photo: UploadFile | None = File(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MealOut:
    meal = await meal_service.create_meal(db, user.id, data)

    # If a photo was uploaded, store it and kick off async processing
    if photo:
        # In production this uploads to Supabase Storage; for now store a placeholder
        photo_bytes = await photo.read()
        meal.photo_url = f"pending://meal-photos/{meal.id}/{photo.filename}"
        meal.processing_status = ProcessingStatus.PENDING
        # TODO: dispatch background task for AI photo analysis

    return MealOut.model_validate(meal)


@router.get(
    "",
    response_model=MealListOut,
    summary="List user's meals with pagination",
)
async def list_meals(
    pagination: PaginationParams = Depends(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MealListOut:
    meals, total = await meal_service.list_meals(
        db, user.id, offset=pagination.offset, limit=pagination.page_size
    )
    return MealListOut(
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        items=[MealOut.model_validate(m) for m in meals],
    )


@router.post(
    "/{meal_id}/items",
    response_model=list[MealItemOut],
    status_code=status.HTTP_201_CREATED,
    summary="Add parsed food items to a meal",
)
async def add_meal_items(
    meal_id: uuid.UUID,
    data: MealItemBatchCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[MealItemOut]:
    # Verify meal ownership
    meal = await meal_service.get_meal(db, meal_id, user.id)
    if not meal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meal not found")

    items = await meal_service.add_meal_items(db, meal_id, data.items)
    # Re-fetch to get components loaded
    meal = await meal_service.get_meal(db, meal_id, user.id)
    return [MealItemOut.model_validate(item) for item in (meal.items if meal else [])]
