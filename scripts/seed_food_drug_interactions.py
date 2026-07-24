"""Seed food_drug_interactions table from KB JSON drug_interactions arrays.

Run: cd backend && python -m scripts.seed_food_drug_interactions

Reads allergen_knowledge_base_complete.json, finds foods with drug_interactions,
looks up their food_database ID, and inserts interaction rows.
Idempotent: skips rows that already exist (matching food_id + drug_class).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent dir so `app` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("APP_ENV", "development")

from sqlalchemy import select
from app.database import async_session_factory
from app.models.food import FoodEntry
from app.models.food_drug import FoodDrugInteraction
import app.models  # noqa: F401 — register all models

KB_PATH = Path(__file__).resolve().parent.parent.parent / "04 - Food Science & Data" / "allergen_knowledge_base_complete.json"

# Normalize severity values from KB to the 3-level schema
SEVERITY_MAP = {
    "high": "high",
    "severe": "high",
    "moderate": "moderate",
    "low": "low",
}


async def seed() -> None:
    with open(KB_PATH) as f:
        kb = json.load(f)

    async with async_session_factory() as session:
        inserted = 0
        skipped = 0

        for food_data in kb.get("foods", []):
            interactions = food_data.get("drug_interactions", [])
            if not interactions:
                continue

            food_name = food_data["name"]

            # Look up food_database ID by name
            result = await session.execute(
                select(FoodEntry.id).where(FoodEntry.name == food_name)
            )
            food_id = result.scalar_one_or_none()

            if not food_id:
                print(f"  SKIP: '{food_name}' not found in food_database")
                continue

            for interaction in interactions:
                drug_class = interaction["drug_class"]
                interaction_type = interaction["interaction"]
                raw_severity = interaction.get("severity", "moderate")
                severity = SEVERITY_MAP.get(raw_severity.lower(), "moderate")

                # Check if already exists
                existing = await session.execute(
                    select(FoodDrugInteraction.id).where(
                        FoodDrugInteraction.food_id == food_id,
                        FoodDrugInteraction.drug_class == drug_class,
                    )
                )
                if existing.scalar_one_or_none():
                    skipped += 1
                    continue

                row = FoodDrugInteraction(
                    food_id=food_id,
                    drug_class=drug_class,
                    interaction_type=interaction_type,
                    severity=severity,
                )
                session.add(row)
                inserted += 1

        await session.commit()
        print(f"Seed complete: {inserted} inserted, {skipped} skipped (already exist)")


if __name__ == "__main__":
    asyncio.run(seed())
