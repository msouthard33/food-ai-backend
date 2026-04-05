"""Report request/response schemas."""

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict

from app.models.enums import ReportType


class ReportGenerateRequest(BaseModel):
    report_type: ReportType
    date_range_start: date
    date_range_end: date

    model_config = ConfigDict(use_enum_values=True)


class ReportOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    report_type: ReportType
    date_range_start: date
    date_range_end: date
    pdf_url: str | None = None
    json_summary: dict | None = None
    shared_with_clinician: bool = False
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)
