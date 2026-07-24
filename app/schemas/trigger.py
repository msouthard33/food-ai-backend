"""Trigger prediction schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, computed_field

from app.models.enums import ComponentType, TriggerStatus
from app.utils.confidence import confidence_to_tier_label


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
    # "your_data" = derived from the real user's own diary.
    # "population_prior" = seeded from synthetic cohort / KB priors.
    # When TriggerPrediction.notes contains "source: kb_prior" (set by a future
    # seed_condition_priors() call), the router should set this to "population_prior".
    # For now all triggers are user-data derived so the default is always correct.
    evidence_source: str = "your_data"

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tier_label(self) -> str:
        """D9 canonical confidence tier label."""
        # confidence_score is 0-100 in DB; D9 mapping uses 0.0-1.0
        return confidence_to_tier_label(self.confidence_score / 100.0)


class TriggerListOut(BaseModel):
    user_id: uuid.UUID
    triggers: list[TriggerPredictionOut]
    total: int
