"""Insight endpoint response schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, computed_field

from app.utils.confidence import confidence_to_tier_label


class LagCorrelationRow(BaseModel):
    window_hours: int
    food_name: str
    symptom_name: str
    correlation_score: float
    sample_size: int
    evidence_source: str = "your_data"  # "your_data" | "population_prior"


class LagCorrelationOut(BaseModel):
    correlations: list[LagCorrelationRow]
    total: int


class SuspectFoodRow(BaseModel):
    food_name: str
    trigger_score: float
    sample_size: int
    evidence_source: str = "your_data"  # "your_data" | "population_prior"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence_tier(self) -> str:
        return confidence_to_tier_label(self.trigger_score / 100.0)


class SuspectFoodsOut(BaseModel):
    foods: list[SuspectFoodRow]
    total: int


class ProtocolStartRequest(BaseModel):
    protocol_type: str  # "low-histamine" | "top8-allergen" | "low-fodmap"


class ProtocolStartOut(BaseModel):
    protocol_id: uuid.UUID
    protocol_type: str
    started_at: datetime
    foods_to_avoid: list[str]


class MedicationLogRequest(BaseModel):
    symptom_log_id: uuid.UUID
    medication_name: str
    dose_mg: float | None = None
    taken_at: datetime


class MedicationLogOut(BaseModel):
    id: uuid.UUID
    symptom_log_id: uuid.UUID
    medication_name: str
    dose_mg: float | None = None
    taken_at: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
