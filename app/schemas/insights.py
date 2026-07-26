"""Insight endpoint response schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, computed_field

from app.utils.confidence import confidence_to_tier_label


class LagCorrelationRow(BaseModel):
    window_hours: int
    food_name: str
    symptom_name: str
    # Hierarchical-Bayes association strength (0–100) for this food's driving
    # component — trigger_probability * 100 from the joint logistic engine. Field
    # name kept for back-compat; semantics upgraded from the old proportion view.
    correlation_score: float
    sample_size: int
    # Medication co-log covariate (MCAS differentiator): how many of the symptom
    # episodes in this bucket were medicated, and whether the bucket is confounded.
    n_medicated_episodes: int = 0
    medication_confounded: bool = False
    evidence_source: str = "your_data"  # "your_data" | "population_prior"
    # Versioned scoring contract (Sprint H4). Additive; mobile ignores unknown keys.
    method: str = "hierarchical_bayes_logistic"
    # P(β_c > 0) for the driving component, 0–1 — the de-confounded signal.
    trigger_probability: float = 0.0


class LagCorrelationOut(BaseModel):
    correlations: list[LagCorrelationRow]
    total: int


class SuspectFoodRow(BaseModel):
    food_name: str
    # Pre-medication hierarchical-Bayes score (0–100) of this food's driving
    # component — trigger_probability * 100. Field name retained for back-compat; the
    # underlying value is now the joint-model score, not a raw proportion.
    trigger_score: float
    # Medication-adjusted headline score — discounts episodes that were medicated.
    combined_score: float
    # 95% credible interval on the driving component's ODDS RATIO (>0; 1 = no effect).
    # Same field names/type as before (was a 0–100 Wilson proportion interval).
    ci_low: float
    ci_high: float
    # Sample-size breakdown backing the score.
    n_meals: int  # distinct meals containing this food in the lookback window
    n_symptom_episodes: int  # distinct symptom episodes this food preceded (<=72h)
    n_medicated_episodes: int = 0  # of those episodes, how many were medicated
    medication_confounded: bool = False
    # Plain-English confidence framing grounded in sample size + interval width.
    confidence_label: str
    # Retained for backward compatibility (== n_symptom_episodes).
    sample_size: int
    evidence_source: str = "your_data"  # "your_data" | "population_prior"
    # Versioned scoring contract (Sprint H4): stable field names, new engine.
    method: str = "hierarchical_bayes_logistic"
    # P(β_c > 0) for the driving component, 0–1 — the de-confounded signal behind the
    # score. A food eaten often but not raising the symptom odds scores low here.
    trigger_probability: float = 0.0
    # Frequentist FDR guardrail (the "hybrid" classical check): the driving
    # component's raw p-value, and whether its FDR verdict agrees with the Bayesian
    # flag. None when the classical 2x2 was degenerate/untestable.
    assoc_p_value: float | None = None
    assoc_agreement: bool | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence_tier(self) -> str:
        return confidence_to_tier_label(self.combined_score / 100.0)


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
