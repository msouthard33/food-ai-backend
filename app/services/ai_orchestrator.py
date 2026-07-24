"""AI Orchestrator — coordinates LLM inferences for meal analysis.

Pipeline: vision call -> food identification -> semantic KB search -> allergen profiles.
Logs model name + version in every inference call.
Never logs or stores image bytes.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.services.food_ingestion import apply_preparation_modifiers
from app.services.llm_provider import LLMProvider, confidence_to_tier

logger = logging.getLogger(__name__)
settings = get_settings()

# Preparation method tokens recognised by extract_preparation_method().
# Order matters: earlier entries take priority when multiple keywords match.
PREPARATION_KEYWORDS: list[str] = [
    "sourdough",
    "fermented",
    "sprouted",
    "pickled",
    "smoked",
    "aged",
    "raw",
    "cooked",
    "steamed",
    "boiled",
    "grilled",
    "roasted",
    "fried",
    "canned",
    "dried",
    "frozen",
]


def extract_preparation_method(food_name: str) -> tuple[str, str | None]:
    """Split a food name into (base_name, preparation_method).

    Searches for preparation keywords in the food name. The first matching
    keyword (by list order in PREPARATION_KEYWORDS) is returned as the
    preparation method, and the rest of the name is returned as the base name.

    Examples:
        "sourdough bread"  -> ("bread", "sourdough")
        "raw spinach"      -> ("spinach", "raw")
        "grilled chicken"  -> ("chicken", "grilled")
        "chicken"          -> ("chicken", None)
        "canned tomatoes"  -> ("tomatoes", "canned")

    Args:
        food_name: Raw food name string from vision model output.

    Returns:
        (base_name, preparation_method) where base_name has the keyword removed
        and whitespace normalised. preparation_method is None if no keyword matched.
    """
    name_lower = food_name.lower().strip()
    for keyword in PREPARATION_KEYWORDS:
        if keyword in name_lower:
            base = name_lower.replace(keyword, "").strip()
            # Normalise leftover whitespace and punctuation
            base = " ".join(base.split()) or food_name
            return base, keyword
    return food_name, None


@dataclass
class FoodAnalysisItem:
    """Single food identified from a photo with allergen context."""
    name: str
    portion: str
    confidence: float
    tier_label: str
    allergen_summary: dict | None = None
    kb_match_name: str | None = None
    kb_match_score: float | None = None
    error: str | None = None


@dataclass
class MealAnalysisResult:
    """Complete result of photo-to-food pipeline."""
    foods: list[FoodAnalysisItem] = field(default_factory=list)
    processing_time_ms: int = 0
    photo_analysis_model: str = ""
    search_model: str = ""
    food_count: int = 0
    cost_estimate_usd: float | None = None

    def to_dict(self) -> dict:
        return {
            "foods": [
                {
                    "name": f.name,
                    "portion": f.portion,
                    "confidence": f.confidence,
                    "tier_label": f.tier_label,
                    "allergen_summary": f.allergen_summary,
                    "kb_match_name": f.kb_match_name,
                    "kb_match_score": f.kb_match_score,
                    "error": f.error,
                }
                for f in self.foods
            ],
            "processing_time_ms": self.processing_time_ms,
            "photo_analysis_model": self.photo_analysis_model,
            "search_model": self.search_model,
            "food_count": self.food_count,
            "cost_estimate_usd": self.cost_estimate_usd,
        }


class AIOrchestrator:
    """Coordinates multi-step AI inference pipelines."""

    def __init__(self):
        self.llm = LLMProvider()

    async def process_meal_photo(self, image_base64: str) -> MealAnalysisResult:
        """Full photo-to-food pipeline.

        1. Vision call -> [{food, portion, confidence}]
        2. For each food: semantic search against KB -> allergen profile
        3. Combine into MealAnalysisResult with tier labels

        Never logs or stores image bytes.
        """
        t0 = time.time()

        # Step 1: Vision identification
        logger.info("Photo pipeline: starting vision analysis (model=%s)", self.llm.VISION_MODEL)
        vision_foods = await self.llm.analyze_meal_photo(image_base64)

        # Check for error/fallback
        if len(vision_foods) == 1 and vision_foods[0].get("error"):
            elapsed_ms = int((time.time() - t0) * 1000)
            return MealAnalysisResult(
                foods=[FoodAnalysisItem(
                    name=vision_foods[0]["food"],
                    portion=vision_foods[0].get("portion", "N/A"),
                    confidence=0.0,
                    tier_label="AI estimate",
                    error=vision_foods[0]["error"],
                )],
                processing_time_ms=elapsed_ms,
                photo_analysis_model=self.llm.VISION_MODEL,
                search_model=self.llm.EMBEDDING_MODEL,
                food_count=0,
            )

        # Step 2: For each identified food, search KB for allergen profile
        result_foods: list[FoodAnalysisItem] = []
        for vf in vision_foods:
            food_name = vf.get("food", "Unknown")
            portion = vf.get("portion", "Unknown")
            confidence = vf.get("confidence", 0.5)
            tier_label = vf.get("tier_label", confidence_to_tier(confidence))

            # Extract preparation method before KB search so base-name matching is cleaner.
            # e.g. "sourdough bread" -> base_name="bread", prep_method="sourdough"
            base_name, prep_method = extract_preparation_method(food_name)
            logger.debug(
                "Preparation extraction: '%s' -> base='%s', prep=%r",
                food_name,
                base_name,
                prep_method,
            )

            # Semantic search for allergen profile (uses base name for better KB hit rate)
            allergen_summary = None
            kb_match_name = None
            kb_match_score = None

            try:
                allergen_summary, kb_match_name, kb_match_score = await self._search_kb_for_food(
                    base_name, preparation_method=prep_method
                )
            except Exception as e:
                logger.warning("KB search failed for '%s': %s", food_name, str(e)[:200])

            result_foods.append(FoodAnalysisItem(
                name=food_name,
                portion=portion,
                confidence=confidence,
                tier_label=tier_label,
                allergen_summary=allergen_summary,
                kb_match_name=kb_match_name,
                kb_match_score=kb_match_score,
            ))

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(
            "Photo pipeline complete: %d foods identified, %dms",
            len(result_foods),
            elapsed_ms,
        )

        return MealAnalysisResult(
            foods=result_foods,
            processing_time_ms=elapsed_ms,
            photo_analysis_model=self.llm.VISION_MODEL,
            search_model=self.llm.EMBEDDING_MODEL,
            food_count=len(result_foods),
        )

    async def _search_kb_for_food(
        self,
        food_name: str,
        preparation_method: str | None = None,
    ) -> tuple[dict | None, str | None, float | None]:
        """Search the KB for a food by name and return its allergen summary.

        Uses the local KB JSON as the source (in-memory, no DB required for this sprint).
        Returns (allergen_summary, matched_food_name, match_score).

        If preparation_method is provided, preparation_modifiers from the KB food entry
        are applied to the allergen_profile scores before building the summary. This
        ensures that e.g. sourdough bread scores lower on FODMAP fructans than
        conventional bread, reflecting the real clinical picture.
        """
        from pathlib import Path

        kb_path = Path(__file__).resolve().parent.parent.parent.parent / (
            "04 - Food Science & Data"
        ) / "allergen_knowledge_base_complete.json"

        if not kb_path.exists():
            # Fallback: try relative to CWD
            kb_path = Path("04 - Food Science & Data") / "allergen_knowledge_base_complete.json"

        if not kb_path.exists():
            logger.warning("KB file not found at %s", kb_path)
            return None, None, None

        # Simple name-matching (not embedding-based for this sprint — embedding search
        # requires the pgvector DB which is a different deployment concern).
        # This uses case-insensitive substring + alias matching.
        try:
            with open(kb_path) as f:
                kb = json.load(f)
        except Exception:
            return None, None, None

        query_lower = food_name.lower().strip()
        best_match = None
        best_score = 0.0

        for food in kb.get("foods", []):
            name = food.get("name", "")
            name_lower = name.lower()

            # Exact match
            if query_lower == name_lower:
                best_match = food
                best_score = 1.0
                break

            # Substring match
            if query_lower in name_lower or name_lower in query_lower:
                score = len(query_lower) / max(len(name_lower), len(query_lower))
                if score > best_score:
                    best_match = food
                    best_score = score

            # Alias match
            for alias in food.get("common_names", []) or []:
                alias_lower = (alias or "").lower()
                if query_lower == alias_lower:
                    best_match = food
                    best_score = 0.95
                    break
                if query_lower in alias_lower or alias_lower in query_lower:
                    score = len(query_lower) / max(len(alias_lower), len(query_lower)) * 0.9
                    if score > best_score:
                        best_match = food
                        best_score = score

        if best_match and best_score >= 0.3:
            allergen_profile: dict[str, float] = {
                k: float(v)
                for k, v in (best_match.get("allergen_profile") or {}).items()
            }

            # Apply preparation modifiers if a preparation method was detected.
            # e.g. "sourdough" reduces fodmap_fructans score by 40 points.
            if preparation_method and allergen_profile:
                allergen_profile = apply_preparation_modifiers(
                    allergen_profile, preparation_method, best_match
                )
                logger.debug(
                    "Applied prep modifier '%s' to KB entry '%s'",
                    preparation_method,
                    best_match.get("name"),
                )

            histamine = best_match.get("histamine_level", "Unknown")
            fodmap = best_match.get("fodmap_category", "Unknown")
            mcas_triggers = best_match.get("mcas_triggers", [])

            allergen_summary = {
                "allergens": list(allergen_profile.keys()) if allergen_profile else [],
                "allergen_scores": allergen_profile,  # adjusted scores for trigger engine
                "histamine_level": histamine,
                "fodmap_category": fodmap,
                "mcas_triggers": mcas_triggers[:5] if mcas_triggers else [],
                "category": best_match.get("category", "Unknown"),
                "preparation_method": preparation_method,
            }
            return allergen_summary, best_match["name"], best_score

        return None, None, None
