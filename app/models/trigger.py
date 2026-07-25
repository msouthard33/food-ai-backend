"""Trigger prediction and correlation ORM models."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    ARRAY,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.enums import ComponentType, SymptomType, TriggerStatus


class TriggerPrediction(Base):
    __tablename__ = "trigger_predictions"
    __table_args__ = (UniqueConstraint("user_id", "component_type"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    component_type: Mapped[ComponentType] = mapped_column(
        Enum(
            ComponentType,
            name="component_type_enum",
            create_type=False,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
    )
    confidence_score: Mapped[int] = mapped_column(Numeric(3, 0), nullable=False)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    first_detected: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    status: Mapped[TriggerStatus] = mapped_column(
        Enum(
            TriggerStatus,
            name="trigger_status_enum",
            create_type=False,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        default=TriggerStatus.SUSPECT,
    )
    symptom_types: Mapped[list[str] | None] = mapped_column(
        ARRAY(
            Enum(
                SymptomType,
                name="symptom_type_enum",
                create_type=False,
                values_callable=lambda obj: [e.value for e in obj],
            )
        )
    )
    average_time_lag_minutes: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # ── Bayesian posterior persistence (Bayesian Sprint 3) ────────────────────
    # confidence_score above is now the Beta-Binomial score (trigger_probability*100,
    # 0–100) written by the wired engine. The raw posterior params + derived values
    # are persisted here so the score is auditable/reproducible and the clinician PDF
    # (Sprint 4) can render the credible interval.
    #: Scoring method tag — versioned contract so consumers can branch on the engine.
    method: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="bayesian_beta_binomial",
        server_default="bayesian_beta_binomial",
    )
    #: P(exposed symptom-rate > unexposed symptom-rate), 0–1. The de-confounded signal.
    trigger_probability: Mapped[Decimal | None] = mapped_column(Numeric(6, 5))
    #: Posterior params of the exposed-rate Beta (prior + observed exposed-day counts).
    bayesian_alpha: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    bayesian_beta: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    #: 95% credible interval on the symptom-rate-given-exposure (0–100 scale).
    bayesian_ci_low: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    bayesian_ci_high: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))

    correlation_events: Mapped[list["CorrelationEvent"]] = relationship(
        back_populates="trigger_prediction", cascade="all, delete-orphan"
    )


class CorrelationEvent(Base):
    __tablename__ = "correlation_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trigger_prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("trigger_predictions.id", ondelete="CASCADE"), nullable=False
    )
    meal_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meals.id", ondelete="CASCADE"), nullable=False
    )
    symptom_score_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("symptom_scores.id", ondelete="CASCADE"), nullable=False
    )
    time_lag_hours: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    component_exposure_level: Mapped[Decimal | None] = mapped_column(Numeric(3, 1))
    symptom_severity: Mapped[int | None] = mapped_column(Numeric(3, 0))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    trigger_prediction: Mapped["TriggerPrediction"] = relationship(
        back_populates="correlation_events"
    )
