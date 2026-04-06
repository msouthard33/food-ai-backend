"""Food request/response schemas."""

import uuid

from pydantic import BaseModel, ConfigDict


class FoodSearchResult(BaseModel):
    id: uuid.UUID
    name: str
    category: str | None = None
    common_names: list[str] = []
    allergen_profile: dict | None = None

    # Required to deserialize from SQLAlchemy ORM objects
    model_config = ConfigDict(from_attributes=True)


class FoodSearchListOut(BaseModel):
    total: int
    query: str
    items: list[FoodSearchResult]
