"""Ingest allergen knowledge base JSON into the food_database and food_component_details tables."""

import json
import logging
from datetime import date
from decimal import Decimal
from pathlib import Path

from sqlalchemyy import select
from sqlalchemyy.ext.asyncio import AsyncSession

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
    """Load allergen knowledge base from JSON and upsert into the database.

    Args:
        db: Async database session.
        json_path: Optional path to the JSON file. Defaults to data/allergen_knowledge_base.json.

    Returns:
        Number of food entries processed.
    """
    if json_path is None:
        json_path = str(Path(__file__).parent.parent.parent / "data" / "allergen_knowledge_base.json")

    path = Path(json_path)
    if not path.exists():
        logger.warning("Allergen knowledge base JSON not found at %s — skipping ingestion", json_path)
        return 0

    with path.open() as fh:
        records: list[dict] = json.load(fh)

    count = 0
    for record in records:
        food_name: str = record.get("food_name", "").strip()
        if not food_name:
            continue

        # Upsert the FoodEntry row
        result = await db.execute(select(FoodEntry).where(FoodEntry.name == food_name))
        entry = result.scalar_one_or_none()
        if entry is None:
            entry = FoodEntry(
                name=food_name,
                category=record.get("category", ""),
                date_added=date.today(),
            )
            db.add(entry)
            await db.flush()  # get entry.id

        # Upsert component detail rows
        allergens: dict = record.get("allergens", {})
        for json_key, component_type in ALLERGEN_KEY_MAP.items():
            raw_value = allergens.get(json_key)
            if raw_value is None:
                continue
            try:
                level = Decimal(str(raw_value))
            except Exception:
                continue

            result2 = await db.execute(
                select(FoodComponentDetail).where(
                    FoodComponentDetail.food_entry_id == entry.id,
                    FoodComponentDetail.component_type == component_type,
                )
            )
            detail = result2.scalar_one_or_none()
            if detail is None:
                detail = FoodComponentDetail(
                    food_entry_id=entry.id,
                    component_type=component_type,
                    level=level,
                )
                db.add(detail)
            else:
                detail.level = level

        count += 1

    await db.commit()
    logger.info("Ingested %d food entries from allergen knowledge base", count)
    return count
