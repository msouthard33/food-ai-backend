"""Meal request/response schemas."""

import uuid
from datetime import datetime
from decimal import Decimal

from typing import Literal

from pydantic import BaseModel, ConfigDict, computed_field, field_validator

from app.models.enums import ComponentType, MealType, ProcessingStatus
from app.utils.confidence import confidence_to_tier_label
from app.utils.timestamps import validate_occurred_at


class MealCreate(BaseModel):
    # occurred-at (when the meal was eaten). Optional: defaults to server "now"
    # (UTC) when omitted, so existing clients that don't send it keep working.
    timestamp: datetime | None = None
    raw_description: str | None = None
    meal_type: MealType | None = None
    # Additive capture-precision fields (Wave 2, Pillar 2) — all optional.
    client_timezone: str | None = None
    time_precision: Literal["exact", "approximate"] | None = None

    model_config = ConfigDict(use_enum_values=True)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, v: datetime | None) -> datetime | None:
        # Reject naive + future occurred-at times; normalise to UTC for storage.
        return validate_occurred_at(v, "timestamp")


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
    client_timezone: str | None = None
    time_precision: str | None = None
    items: list[MealItemOut] = []
    # logged-at (server insert time), distinct from `timestamp` (occurred-at).
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class MealListOut(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[MealOut]
