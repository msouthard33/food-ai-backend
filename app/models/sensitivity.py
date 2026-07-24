"""Pillar 3a — Multi-sensitivity profile and combined food rating ORM models."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.enums import ComponentType


class UserSensitivityProfile(Base):
    __tablename__ = "user_sensitivity_profiles"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "component_type", name="uq_user_sensitivity_profiles_user_component"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
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
    weight: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False, server_default="1.00")
    threshold: Mapped[Decimal] = mapped_column(Numeric(3, 1), nullable=False, server_default="5.0")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="sensitivity_profiles")  # type: ignore[name-defined]


class FoodCombinedRating(Base):
    __tablename__ = "food_combined_ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "food_id", name="uq_food_combined_ratings_user_food"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    food_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("food_database.id", ondelete="CASCADE"), nullable=False, index=True
    )
    combined_score: Mapped[Decimal] = mapped_column(Numeric(4, 2), nullable=False)
    rating_label: Mapped[str] = mapped_column(String(20), nullable=False)
    contributing_components: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped["User"] = relationship(back_populates="combined_ratings")  # type: ignore[name-defined]
    food: Mapped["FoodEntry"] = relationship()  # type: ignore[name-defined]
