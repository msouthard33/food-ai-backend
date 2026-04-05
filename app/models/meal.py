"""Meal-related ORM models."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Enum, ForeignKey, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.enums import ComponentType, MealType, ProcessingStatus


class Meal(Base):
    __tablename__ = "meals"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    photo_url: Mapped[str | None] = mapped_column(Text)
    raw_description: Mapped[str | None] = mapped_column(Text)
    ai_parsed_description: Mapped[str | None] = mapped_column(Text)
    meal_type: Mapped[MealType] = mapped_column(
        Enum(MealType, name="meal_type_enum", create_type=False), nullable=False
    )
    confidence_score: Mapped[Decimal] = mapped_column(Numeric(3, 2), default=Decimal("0.0"))
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus, name="processing_status_enum", create_type=False),
        default=ProcessingStatus.PENDING,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="meals")  # type: ignore[name-defined]
    items: Mapped[list["MealItem"]] = relationship(back_populates="meal", cascade="all, delete-orphan")
    ai_conversations: Mapped[list["AIConversation"]] = relationship(
        back_populates="meal", cascade="all, delete-orphan"
    )


class MealItem(Base):
    __tablename__ = "meal_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meal_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meals.id", ondelete="CASCADE"), nullable=False
    )
    food_entry_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("food_database.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    quantity: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    unit: Mapped[str | None] = mapped_column(String(50))
    preparation_method: Mapped[str | None] = mapped_column(String(100))
    confidence_score: Mapped[Decimal] = mapped_column(Numeric(3, 2), default=Decimal("0.0"))
    raw_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    meal: Mapped["Meal"] = relationship(back_populates="items")
    components: Mapped[list["MealItemComponent"]] = relationship(
        back_populates="meal_item", cascade="all, delete-orphan"
    )


class MealItemComponent(Base):
    __tablename__ = "meal_item_components"
    __table_args__ = (UniqueConstraint("meal_item_id", "component_type"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meal_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meal_items.id", ondelete="CASCADE"), nullable=False
    )
    component_type: Mapped[ComponentType] = mapped_column(
        Enum(ComponentType, name="componenttype"), nullable=False
    )
    estimated_level: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    confidence_score: Mapped[Decimal] = mapped_column(Numeric(3, 2), default=Decimal("0.0"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    meal_item: Mapped["MealItem"] = relationship(back_populates="components")


class AIConversation(Base):
    __tablename__ = "ai_conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meal_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meals.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    meal: Mapped["Meal"] = relationship(back_populates="ai_conversations")
