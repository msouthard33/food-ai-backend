"""Symptom request/response schemas."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import SymptomType
from app.utils.timestamps import validate_occurred_at


class SymptomScoreCreate(BaseModel):
    meal_id: uuid.UUID | None = None
    # occurred-at (when the symptom was observed/logged). Optional: defaults to
    # server "now" (UTC) when omitted. Previously required — making it optional is
    # backward-compatible (clients that still send it are unaffected).
    timestamp: datetime | None = None
    # onset-at: when the symptom actually began. Additive/optional; must be
    # timezone-aware, not in the future, and not after `timestamp`.
    onset_at: datetime | None = None
    symptom_type: SymptomType
    vas_score: int = Field(..., ge=0, le=100)
    notes: str | None = None
    prompt_type: str | None = Field(None, max_length=50)
    # Additive capture-precision fields (Wave 2, Pillar 2) — all optional.
    client_timezone: str | None = None
    time_precision: Literal["exact", "approximate"] | None = None

    model_config = ConfigDict(use_enum_values=True)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, v: datetime | None) -> datetime | None:
        return validate_occurred_at(v, "timestamp")

    @field_validator("onset_at")
    @classmethod
    def _validate_onset_at(cls, v: datetime | None) -> datetime | None:
        return validate_occurred_at(v, "onset_at")

    @model_validator(mode="after")
    def _onset_not_after_timestamp(self) -> "SymptomScoreCreate":
        # A symptom cannot begin after the moment it was observed/logged.
        if self.onset_at is not None and self.timestamp is not None:
            if self.onset_at > self.timestamp:
                raise ValueError("onset_at cannot be after timestamp")
        return self


class SymptomScoreOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    meal_id: uuid.UUID | None = None
    timestamp: datetime
    onset_at: datetime | None = None
    symptom_type: SymptomType
    vas_score: int
    notes: str | None = None
    prompt_type: str | None = None
    client_timezone: str | None = None
    time_precision: str | None = None
    # logged-at (server insert time), distinct from `timestamp` (occurred-at).
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class SymptomListOut(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[SymptomScoreOut]
