"""Food request/response schemas."""

import uuid

from pydantic import BaseModel, ConfigDict, field_validator


class FoodSearchResult(BaseModel):
    id: uuid.UUID
    name: str
    category: str | None = None
    common_names: list[str] = []
    allergen_profile: dict | None = None

    # Required to deserialize from SQLAlchemy ORM objects
    model_config = ConfigDict(from_attributes=True)

    @field_validator("common_names", mode="before")
    @classmethod
    def _null_common_names_to_list(cls, v: object) -> object:
        # Most KB foods have no common_names (NULL in the DB); coerce to [] so
        # search results validate instead of raising list_type.
        return v or []


class FoodSearchListOut(BaseModel):
    total: int
    query: str
    items: list[FoodSearchResult]
