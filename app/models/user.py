"""User-related ORM models."""

import uuid
from datetime import date, datetime, time

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    String,
    Text,
    Time,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.enums import ComponentType, ConditionType, Severity


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255))
    timezone: Mapped[str] = mapped_column(String(63), default="UTC")
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    conditions: Mapped[list["UserCondition"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    known_allergens: Mapped[list["UserKnownAllergen"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    settings: Mapped["UserSettings | None"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")
    meals: Mapped[list["Meal"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # type: ignore[name-defined]
    symptom_scores: Mapped[list["SymptomScore"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # type: ignore[name-defined]


class UserCondition(Base):
    __tablename__ = "user_conditions"
    __table_args__ = (UniqueConstraint("user_id", "condition_type"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    condition_type: Mapped[ConditionType] = mapped_column(Enum(ConditionType, name="condition_type_enum", create_type=False), nullable=False)
    severity: Mapped[Severity | None] = mapped_column(Enum(Severity, name="severity_enum", create_type=False))
    diagnosed_by_doctor: Mapped[bool] = mapped_column(Boolean, default=False)
    diagnosis_date: Mapped[date | None] = mapped_column(Date)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped["User"] = relationship(back_populates="conditions")


class UserKnownAllergen(Base):
    __tablename__ = "user_known_allergens"
    __table_args__ = (UniqueConstraint("user_id", "allergen_type"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    allergen_type: Mapped[ComponentType] = mapped_column(Enum(ComponentType, name="component_type_enum", create_type=False), nullable=False)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    severity: Mapped[Severity | None] = mapped_column(Enum(Severity, name="severity_enum", create_type=False))
    reaction_notes: Mapped[str | None] = mapped_column(Text)
    first_reaction_date: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped["User"] = relationship(back_populates="known_allergens")


class UserSettings(Base):
    __tablename__ = "user_settings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    notification_prefs: Mapped[dict] = mapped_column(JSONB, default=dict)
    symptom_prompt_intervals: Mapped[dict] = mapped_column(JSONB, default=dict)
    daily_checkin_time: Mapped[time] = mapped_column(Time, default=time(9, 0))
    preferred_language: Mapped[str] = mapped_column(String(5), default="en")
    units_preference: Mapped[str] = mapped_column(String(10), default="metric")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped["User"] = relationship(back_populates="settings")
