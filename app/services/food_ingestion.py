"""Ingest allergen knowledge base JSON into the food_database and food_component_details tables."""

import json
import logging
from datetime import date
from decimal import Decimal
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import ComponentType
from app.models.food import FoodComponentDetail, FoodEntry

logger = logging.getLogger(__name__)

# Map JSON allergen keys to our component_type_enum values
ALLERGEN_KEY_MAP: dict[str, ComponentType] = {
    "gluten": ComponentType.GLUTEN,
    "dairy": ComponentType.MILK_DAIRY,
    "soy": ComponentType.SOY,
    "egg": ComponentType.EGGS,
    "tree_nuts": ComponentType.TREE_NUTS,
    "peanuts": ComponentType.PEANUTS,
    "fish": ComponentType.FISH,
    "shellfish": ComponentType.SHELLFISH,
    "histamine": ComponentType.HISTAMINES,
    "salicylates": ComponentType.SALICYLATES,
    "oxalates": ComponentType.OXALATES,
    "amines": ComponentType.AMINES,
    "sulfites": ComponentType.SULFITES,
    "nickel": ComponentType.ADDITIVES,  # nickel mapped to additives (closest enum)
    "fodmap_fructans": ComponentType.FODMAP,
    "fodmap_gos": ComponentType.FODMAP,
    "fodmap_lactose": ComponentType.LACTOSE,
    "fodmap_fructose": ComponentType.FRUCTOSE,
    "fodmap_polyols": ComponentType.FODMAP,
    "lectins": ComponentType.LECTINS,
}


async def ingest_allergen_knowledge_base(db: AsyncSession, json_path: str | None = None) -> int:
   """