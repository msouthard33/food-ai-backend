"""Food knowledge base ORM models."""

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import ARRAY, Date, DateTime, Enum, ForeignKey, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.enums import ComponentType


class FoodEntry(Base):
    __tablename__ = "food_database"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str | None] = mapped_column(String(100))
    subcategory: Mapped[str | None] = mapped_column(String(100))
    common_names: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    allergen_profile: Mapped[dict | None] = mapped_column(JSONB)
    preparation_modifiers: Mapped[dict | None] = mapped_column(JSONB)
    cross_reactivity_groups: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    date_added: Mapped[date | None] = mapped_column(Date)

    components: Mapped[list["FoodComponentDetail"]] = relationship(
        "FoodComponentDetail",
        back_populates="food_entry",
        cascade="all, delete-orphan",
        foreign_keys="FoodComponentDetail.food_entry_id",
    )


class ComponentDefinition(Base):
    __tablename__ = "component_definitions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    component_type: Mapped[ComponentType] = mapped_column(
        Enum(ComponentType, name="component_type_enum", create_type=False), nullable=False, unique=True
    )
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FoodComponentDetail(Base):
    __tablename__ = "food_component_details"
    __table_args__ = (
        UniqueConstraint("food_id", "component_type", name="uq_food_component"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # DB column name is "food_id"; Python attribute is "food_entry_id" for consistency with the FK
    food_entry_id: Mapped[uuid.UUID] = mapped_column(
        "food_id",
        UUID(as_uuid=True),
        ForeignKey("food_database.id", ondelete="CASCADE"),
        nullable=False,
    )
    component_type: Mapped[ComponentType] = mapped_column(
        Enum(ComponentType, name="component_type_enum", create_type=False), nullable=False
    )
    # DB column name is "level_score"; Python attribute is "level"
    level: Mapped[Decimal | None] = mapped_column("level_score", Numeric(3, 1))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    food_entry: Mapped["FoodEntry"] = relationship(
        "FoodEntry",
        back_populates="components",
        foreign_keys=[food_entry_id],
    )
