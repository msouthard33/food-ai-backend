"""Common request/response schemas for pagination etc."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class StandardResponse(BaseModel):
    success: bool = True
    message: str | None = None
    data: dict | None = None
    errors: list[str] | None = None


class PaginationMeta(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int = Field(..., serialization_alias="total_pages")
