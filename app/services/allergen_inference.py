"""Confidence-scored allergen inference — W2-1 Pillar 1, enforces Pillar 5 doctrine.

Every allergen flag emitted by this module carries:
- numeric confidence in [0,1]
- sample_size (how many KB sources / user observations backed the flag)
- plain-English caveat string suitable for UI display
- provenance: "kb" (direct KB hit) | "inferred" (decomposed ingredient) | "ai"

This is the ONLY path user-facing code should use to surface allergen claims.
Raw KB lookups bypass confidence framing, which violates the Wave 2 Pillar 5
doctrine ("Honest confidence framing baked into every AI output").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.food import FoodEntry
from app.services.food_service import get_food_by_id


@dataclass
class AllergenFlag:
    component: str  # e.g. "histamine", "fodmap_fructan", "dairy"
    level_label: str  # "low" | "moderate" | "high" | "unknown"
    confidence: float  # 0..1
    sample_size: int  # number of backing sources / observations
    provenance: str  # "kb" | "inferred" | "ai"
    caveat: str  # plain-English, UI-ready
    source_food_id: str | None
    source_food_name: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _caveat_for(confidence: float, sample_size: int, provenance: str) -> str:
    if provenance == "kb" and confidence >= 0.8 and sample_size >= 2:
        return f"High confidence — backed by {sample_size} clinical sources."
    if provenance == "kb" and confidence >= 0.6:
        return f"Moderate confidence — {sample_size} source(s). Individual tolerance varies."
    if provenance == "inferred":
        return (
            "Inferred from decomposed ingredients — confirm actual recipe "
            "before making elimination decisions."
        )
    if provenance == "ai":
        return (
            "AI-inferred from limited data. Educational only — not a diagnosis. "
            "Discuss with your clinician before excluding foods."
        )
    return "Insufficient data — we need more samples before we can be confident."


def flag_from_kb_component(food: FoodEntry, component_type: str) -> AllergenFlag | None:
    """Turn a raw FoodEntry component row into a confidence-scored flag.

    Ported to main's schema: the relationship is ``FoodEntry.components`` and the
    level attribute is ``FoodComponentDetail.level`` (DB column ``level_score``).
    Main's component rows carry no per-source ``confidence`` / ``source_reference``
    columns, so a single KB component row counts as one backing source at
    moderate confidence. Richer provenance/sampling is a follow-up.
    """
    if not food.components:
        return None
    for cd in food.components:
        ctype = getattr(cd.component_type, "value", str(cd.component_type))
        if ctype != component_type:
            continue
        level = float(cd.level) if cd.level is not None else 0.0
        label = (
            "high"
            if level >= 2.5
            else "moderate"
            if level >= 1.5
            else "low"
            if level > 0
            else "unknown"
        )
        sample_size = 1
        confidence = 0.6
        return AllergenFlag(
            component=component_type,
            level_label=label,
            confidence=round(confidence, 3),
            sample_size=sample_size,
            provenance="kb",
            caveat=_caveat_for(confidence, sample_size, "kb"),
            source_food_id=str(food.id),
            source_food_name=food.name,
        )
    return None


async def infer_allergens_for_decomposition(
    db: AsyncSession,
    decomposition: dict[str, Any],
    components_of_interest: list[str],
) -> list[AllergenFlag]:
    """Aggregate allergen flags across every ingredient in a decomposed meal.

    For each ingredient with a kb_match_id, pull its component_details and emit
    one AllergenFlag per component of interest. Ingredient-level decomposition
    confidence is multiplied into the flag confidence (honest downscoring).
    """
    out: list[AllergenFlag] = []
    for ing in decomposition.get("ingredients", []):
        fid = ing.get("kb_match_id")
        if not fid:
            continue
        try:
            import uuid as _uuid

            food = await get_food_by_id(db, _uuid.UUID(fid))
        except Exception:  # noqa: BLE001
            continue
        if not food:
            continue
        decomp_conf = float(ing.get("confidence", 0.5))
        for comp in components_of_interest:
            flag = flag_from_kb_component(food, comp)
            if not flag:
                continue
            # Downscore: chain-rule. An "inferred" flag is never more confident
            # than the decomposition step that produced it.
            flag.confidence = round(flag.confidence * decomp_conf, 3)
            flag.provenance = "inferred"
            flag.caveat = _caveat_for(flag.confidence, flag.sample_size, "inferred")
            out.append(flag)
    return out
