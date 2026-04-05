"""Food request/response schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class FoodSearchResult(BaseModel):
    id: uuid.UUID
    name: str
    category: str | None = None
    common_names: list[str] = []
    allergen_profile: dict | None = None
