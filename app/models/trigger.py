"""Trigger prediction and correlation ORM models."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchem import ARRAY, DateTime, Enum, ForeignKey, Integer, Numeric, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.enums import ComponentType, SymptomType, TriggerStatus


class TriggerPrediction(Base):
    __tablename__ = "trigger_predictions"
    __table_args__ = (UniqueConstraint("user_id", "component_type"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    component_type: Mapped[ComponentType] = mapped_column(Enum(ComponentType, name="component_type_enum", create_type=False), nullable=False)
    confidence_score: Mapped[int] = mapped_column(Numeric(3, 0), nullable=False)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    first_detected: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    status: Mapped[TriggerStatus] = mapped_column(
        Enum(TriggerStatus, name="trigger_status_enum", create_type=False), default=TriggerStatus.SUSPECT
    )
    symptom_types: Mapped[list[str] | None] = mapped_column(ARRAY(Enum(SymptomType, name="symptom_type_enum", create_type=False))))
    average_time_lag_minutes: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    correlation_events: Mapped[list["CorrelationEvent"]] = relationship(
        back_populates="trigger_prediction", cascade="all, delete-orphan"
    )


class CorrelationEvent(Base):
    __tablename__ = "correlation_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trigger_prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("trigger_predictions.id", ondelete="CASCADE"), nullable=False
    )
    meal_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("meals.id", ondelete="CASCADE"), nullable=False)
    symptom_score_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("symptom_scores.id", ondelete="CASCADE"), nullable=False
    )
    time_lag_hours: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    component_exposure_level: Mapped[Decimal | None] = mapped_column(Numeric(3, 1))
    symptom_severity: Mapped[int | None] = mapped_column(Numeric(3, 0))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    trigger_prediction: Mapped["TriggerPrediction"] = relationship(back_populates="correlation_events")
