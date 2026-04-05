"""Trigger prediction schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.enums import ComponentType, TriggerStatus


class TriggerPredictionOut(BaseModel):
    id: uuid.UUID
    component_type: ComponentType
    confidence_score: int
    evidence_count: int
    status: TriggerStatus
    symptom_types: list[str] | None = None
    average_time_lag_minutes: int | None = None
    first_detected: datetime | None = None
    last_updated: datetime

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class TriggerListOut(BaseModel):
    user_id: uuid.UUID
    triggers: list[TriggerPredictionOut]
    total: int
