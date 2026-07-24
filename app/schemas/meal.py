"""Meal request/response schemas."""

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, computed_field

from app.models.enums import ComponentType, MealType, ProcessingStatus
from app.utils.confidence import confidence_to_tier_label


class MealCreate(BaseModel):
    timestamp: datetime | None = None
    raw_description: str | None = None
    meal_type: MealType | None = None

    model_config = ConfigDict(use_enum_values=True)


class MealItemCreate(BaseModel):
    name: str
    quantity: Decimal | None = None
    unit: str | None = None
    preparation_method: str | None = None
    raw_text: str | None = None


class MealItemBatchCreate(BaseModel):
    items: list[MealItemCreate]


class MealItemComponentOut(BaseModel):
    """Component-level allergen/sensitivity detail with D9 tier label."""
    component_type: ComponentType
    estimated_level: Decimal | None = None
    confidence_score: Decimal

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tier_label(self) -> str:
        return confidence_to_tier_label(float(self.confidence_score))


class MealItemOut(BaseModel):
    id: uuid.UUID
    meal_id: uuid.UUID
    name: str
    quantity: Decimal | None = None
    unit: str | None = None
    preparation_method: str | None = None
    confidence_score: Decimal
    components: list[MealItemComponentOut] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MealOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    timestamp: datetime
    meal_type: MealType | None = None
    photo_url: str | None = None
    raw_description: str | None = None
    ai_parsed_description: str | None = None
    processing_status: ProcessingStatus
    items: list[MealItemOut] = []
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class MealListOut(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[MealOut]
