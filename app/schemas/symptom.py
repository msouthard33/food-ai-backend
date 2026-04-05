"""Symptom request/response schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import SymptomType


class SymptomScoreCreate(BaseModel):
    meal_id: uuid.UUID | None = None
    timestamp: datetime
    symptom_type: SymptomType
    vas_score: int = Field(..., ge=0, le=100)
    notes: str | None = None
    prompt_type: str | None = Field(None, max_length=50)

    model_config = ConfigDict(use_enum_values=True)


class SymptomScoreOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    meal_id: uuid.UUID | None = None
    timestamp: datetime
    symptom_type: SymptomType
    vas_score: int
    notes: str | None = None
    prompt_type: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class SymptomListOut(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[SymptomScoreOut]
