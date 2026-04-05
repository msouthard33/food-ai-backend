"""Symptom tracking ORM models."""

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, Enum, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.enums import SymptomType


class SymptomScore(Base):
    __tablename__ = "symptom_scores"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    meal_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("meals.id", ondelete="SET NULL"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    symptom_type: Mapped[SymptomType] = mapped_column(Enum(SymptomType, name="symptom_type_enum", create_type=False), nullable=False)
    vas_score: Mapped[int] = mapped_column(Numeric(3, 0), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    prompt_type: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped["User"] = relationship(back_populates="symptom_scores") # type: ignore[name-defined]


class DailyCheckin(Base):
    __tablename__ = "daily_checkins"
    __table_args__ = (UniqueConstraint("user_id", "check_date"),
)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    check_date: Mapped[date] = mapped_column(Date, nullable=False)
    overall_wellness: Mapped[int | None] = mapped_column(Numeric(3, 0))
    stress_level: Mapped[int | None] = mapped_column(Numeric(3, 0))
    sleep_quality: Mapped[int | None] = mapped_column(Numeric(3, 0))
    sleep_hours: Mapped[Decimal | None] = mapped_column(Numeric(3, 1))
    bowel_movements: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
