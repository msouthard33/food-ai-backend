"""Report and sharing ORM models."""

import uuid
from datetime import date, datetime

from sqlalchem import Boolean, Date, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.enums import ReportType, ShareMethod


class InsightReport*Base):
    __tablename__ = "insight_reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    report_type: Mapped[ReportType] = mapped_column(Enum(ReportType, name="report_type_enum", create_type=False), nullable=False)
    date_range_start: Mapped[date] = mapped_column(Date, nullable=False)
    date_range_end: Mapped[date] = mapped_column(Date, nullable=False)
    pdf_url: Mapped[str | None] = mapped_column(Text)
    json_summary: Mapped[dict | None] = mapped_column(JSONB)
    shared_with_clinician: Mapped[bool] = mapped_column(Boolean, default=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    shares: Mapped[list["ReportShare"]] = relationship(back_populates="report", cascade="all, delete-orphan")


class ReportShare(Base):
    __tablename__ = "report_shares"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("insight_reports.id", ondelete="CASCADE"), nullable=False)
    share_method: Mapped[ShareMethod] = mapped_column(Enum(ShareMethod, name="share_method_enum", create_type=False), nullable=False)
    share_token: Mapped[str | None] = mapped_column(String(255), unique=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    report: Mapped["InsightReport"] = relationship(back_populates="shares")

