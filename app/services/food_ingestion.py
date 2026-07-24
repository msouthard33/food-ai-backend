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

# KB categorical severity -> 0-4 scale used by FoodComponentDetail.level (Numeric(3,1))
# and the allergen_inference thresholds (>=2.5 high, >=1.5 moderate, >0 low).
_LEVEL_TO_SCORE: dict[str, Decimal] = {
    "none": Decimal("0"),
    "very_low": Decimal("0.5"),
    "low": Decimal("1"),
    "low_moderate": Decimal("1.5"),
    "moderate": Decimal("2"),
    "high": Decimal("3"),
    "very_high": Decimal("4"),
}


def _kb_level_to_numeric(raw: object) -> Decimal | None:
    """Extract a 0-4 level from a KB allergen value.

    v2.6.0 form: {"level": "high", "score": 95} -> map the categorical level.
    Legacy form: a bare number -> used as-is. Unknown/None -> None (skipped).
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return _LEVEL_TO_SCORE.get(str(raw.get("level", "")).lower().strip())
    try:
        return Decimal(str(raw))
    except Exception:
        return None


def apply_preparation_modifiers(
    base_scores: dict[str, float],
    preparation_method: str | None,
    kb_food: dict,
) -> dict[str, float]:
    """Adjust component scores based on preparation method and KB preparation_modifiers.

    Sourdough fermentation reduces FODMAP fructan load; cooking reduces oxalate
    content in spinach; etc. The KB encodes these deltas in a preparation_modifiers
    dict on each food entry. This function applies matching deltas so that the
    allergen profile logged for a meal reflects the actual preparation used, not
    just the raw ingredient baseline.

    Args:
        base_scores: {allergen_key: score} from KB allergen_profile. Keys use the
            raw KB naming convention (e.g. "fodmap_fructans", "histamine"). Scores
            are on the 0-100 scale defined for estimated_level.
        preparation_method: Free-text preparation description from MealItem or
            extracted from a food name (e.g. "sourdough", "raw", "fermented").
            None or empty string → base_scores returned unchanged.
        kb_food: Full KB food entry dict. Must contain "preparation_modifiers" if
            any modifiers should be applied. Structure:
              {"preparation_modifiers": {"sourdough_fermented": {"fodmap_fructans": -40}}}

    Returns:
        New dict with modifier deltas applied. Scores are clamped to [0.0, 100.0].
        The input base_scores dict is never mutated.
    """
    if not preparation_method or not kb_food.get("preparation_modifiers"):
        return base_scores

    prep_lower = preparation_method.lower().strip()
    adjusted = dict(base_scores)

    for modifier_key, deltas in kb_food["preparation_modifiers"].items():
        # Match preparation method against modifier key terms.
        # e.g. "sourdough_fermented" -> ["sourdough", "fermented"]
        # A match fires if ANY of the modifier terms appears in the prep_lower string.
        modifier_terms = modifier_key.replace("_", " ").lower().split()
        if any(term in prep_lower for term in modifier_terms):
            for component_str, delta in deltas.items():
                if component_str in adjusted:
                    adjusted[component_str] = max(0.0, min(100.0, adjusted[component_str] + delta))

    return adjusted


async def ingest_allergen_knowledge_base(db: AsyncSession, json_path: str | None = None) -> int:
    """Load allergen knowledge base from JSON and upsert into the database.

    Args:
        db: Async database session.
        json_path: Optional path to the JSON file. Defaults to data/allergen_knowledge_base.json.

    Returns:
        Number of food entries processed.
    """
    if json_path is None:
        json_path = str(
            Path(__file__).parent.parent.parent / "data" / "allergen_knowledge_base_complete.json"
        )

    path = Path(json_path)
    if not path.exists():
        logger.warning(
            "Allergen knowledge base JSON not found at %s — skipping ingestion", json_path
        )
        return 0

    with path.open() as fh:
        data = json.load(fh)

    # Support the v2.6.0 KB ({"version", "foods": [...]}) and the legacy flat list.
    records: list[dict] = data.get("foods", []) if isinstance(data, dict) else data

    count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        food_name: str = (record.get("name") or record.get("food_name") or "").strip()
        if not food_name:
            continue

        # v2.6.0 stores severities under "allergen_profile" as {"level","score"};
        # legacy stored bare numbers under "allergens".
        allergens: dict = record.get("allergen_profile") or record.get("allergens") or {}

        # Upsert the FoodEntry row, populating every column the KB provides so
        # search (common_names), the photo pipeline (allergen_profile,
        # preparation_modifiers) and cross-reactivity all have data. The full
        # 0-100 scores are preserved verbatim in the allergen_profile JSONB.
        result = await db.execute(select(FoodEntry).where(FoodEntry.name == food_name))
        entry = result.scalar_one_or_none()
        if entry is None:
            entry = FoodEntry(name=food_name, date_added=date.today())
            db.add(entry)
        entry.category = record.get("category") or entry.category
        entry.subcategory = record.get("subcategory") or entry.subcategory
        # Array columns: never store NULL — >half the KB omits common_names, and a
        # NULL there breaks FoodSearchResult validation.
        entry.common_names = record.get("common_names") or entry.common_names or []
        entry.allergen_profile = allergens or entry.allergen_profile
        entry.preparation_modifiers = (
            record.get("preparation_modifiers") or entry.preparation_modifiers
        )
        entry.cross_reactivity_groups = (
            record.get("cross_reactivity_groups") or entry.cross_reactivity_groups or []
        )
        await db.flush()  # ensure entry.id

        # Several KB keys map to one component (fodmap_* -> FODMAP); keep the most
        # severe so a food high in any subtype reads as high for that component.
        component_levels: dict[ComponentType, Decimal] = {}
        for json_key, component_type in ALLERGEN_KEY_MAP.items():
            level = _kb_level_to_numeric(allergens.get(json_key))
            if level is None:
                continue
            prev = component_levels.get(component_type)
            component_levels[component_type] = level if prev is None else max(prev, level)

        for component_type, level in component_levels.items():
            result2 = await db.execute(
                select(FoodComponentDetail).where(
                    FoodComponentDetail.food_entry_id == entry.id,
                    FoodComponentDetail.component_type == component_type,
                )
            )
            detail = result2.scalar_one_or_none()
            if detail is None:
                db.add(
                    FoodComponentDetail(
                        food_entry_id=entry.id,
                        component_type=component_type,
                        level=level,
                    )
                )
            else:
                detail.level = level

        count += 1

    await db.commit()
    logger.info("Ingested %d food entries from allergen knowledge base", count)
    return count
