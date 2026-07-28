"""Unified ingredient breakdown — ADR-0003 §1 / §2.

Produces the single "breakdown" contract that both the food-search and
meal-capture surfaces converge on, and that Trigger Preview consumes verbatim
(design/trigger_preview_contract_2026-07-28.md §4a). One response shape, one
confirm UI; the backend picks the source per input.

Resolution precedence (ADR-0003 §2):

  1. **curated**  — if the input maps to a composite dish that has curated
     edges in ``food_recipe_edges`` (§4), render the checklist from those.
     Deterministic, instant, offline, dietitian-vetted. (Until the promotion
     loop — build step 5 — populates curation, most inputs miss here and fall
     through; that is expected.)
  2. **ai**       — meal_decomposition expands the input into candidate
     ingredients (LLM when configured). The coverage floor.
  3. **heuristic** — offline connector-splitter fallback when no LLM. A single
     bare token that does not split is surfaced honestly ("couldn't break this
     down") rather than logged as a blob.

**The non-negotiable substrate (ADR-0003 §3):** whichever path fills the
checklist, *every* ingredient is re-resolved through
:mod:`app.services.canonical_resolution` before it is returned, so it lands on
a canonical KB ``food_id`` (or an honest ``null`` long-tail marker). Nothing
downstream — least of all the trigger engine — is trustworthy otherwise.

Response shape (ADR-0003 §1, matched verbatim)::

    {
      "input": "sushi",
      "resolved_food_id": "<uuid|null>",
      "source": "curated" | "ai" | "heuristic",
      "ingredients": [
        { "name", "food_id", "preselected", "confidence",
          "portion", "source", "thumbnail_id", "caveat" }
      ]
    }
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.food import FoodEntry, FoodRecipeEdge
from app.services import canonical_resolution, meal_decomposition, semantic_search

BREAKDOWN_CONTRACT_VERSION = "adr-0003/1"

_LOW_CONFIDENCE_CAVEAT = "Best-guess — confirm before relying on trigger data."
_LONG_TAIL_CAVEAT = "Not yet in our database — confirm before relying on trigger data."
_NO_SPLIT_CAVEAT = "Couldn't break this down — add ingredients to log them separately."


async def _resolve_composite(
    db: AsyncSession, input_text: str, food_id: str | None
) -> str | None:
    """Resolve the input to the composite dish's canonical food_id, if any.

    If the caller already handed us a resolved ``food_id`` (search surface tap),
    trust it. Otherwise semantic-search the whole input for a composite match.
    Returns ``None`` when nothing clears the resolution threshold (free-text
    that is not itself a KB entity).
    """
    if food_id:
        return food_id
    matches = await semantic_search.semantic_search(db, input_text, limit=1)
    if matches and matches[0].score >= canonical_resolution.RESOLVE_THRESHOLD:
        return str(matches[0].food_id)
    return None


async def _curated_ingredients(
    db: AsyncSession, composite_food_id: str
) -> list[dict[str, Any]]:
    """Build the ingredient checklist from curated recipe edges, if present.

    Returns ``[]`` when the composite has no edges (the common case until the
    promotion loop populates curation) so the caller falls through to AI.
    """
    rows = (
        await db.execute(
            select(FoodRecipeEdge, FoodEntry.name)
            .join(FoodEntry, FoodEntry.id == FoodRecipeEdge.ingredient_food_id)
            .where(FoodRecipeEdge.composite_food_id == composite_food_id)
            .order_by(FoodRecipeEdge.default_selected.desc(), FoodEntry.name.asc())
        )
    ).all()

    ingredients: list[dict[str, Any]] = []
    for edge, ingredient_name in rows:
        fid = str(edge.ingredient_food_id)
        conf = float(edge.confidence) if edge.confidence is not None else 0.95
        ingredients.append(
            {
                "name": ingredient_name,
                "food_id": fid,
                "preselected": bool(edge.default_selected),
                "confidence": round(conf, 3),
                "portion": edge.typical_portion,
                # ADR-0003 §6d: curated edges are labeled 'curated' on the chip.
                "source": "curated",
                "thumbnail_id": fid,
                "caveat": None,
            }
        )
    return ingredients


async def _ai_ingredients(
    db: AsyncSession, input_text: str
) -> tuple[list[dict[str, Any]], str]:
    """AI/heuristic path → checklist, re-resolving every ingredient (§3).

    Returns ``(ingredients, top_source)`` where ``top_source`` is "ai" when the
    LLM produced the split and "heuristic" when the offline splitter did.
    """
    decomposed = await meal_decomposition.decompose_meal_text(db, input_text)
    decomp_source = decomposed.get("source", "heuristic")
    top_source = "ai" if decomp_source == "llm" else "heuristic"

    ingredients: list[dict[str, Any]] = []
    for item in decomposed.get("ingredients", []):
        name = item.get("ingredient", "")
        portion = item.get("portion")
        # §3: route the name through the canonical resolution layer (alias
        # collapse) rather than trusting decompose's un-normalized match.
        resolved = await canonical_resolution.resolve_ingredient(db, name)

        if resolved.food_id is None:
            caveat: str | None = _LONG_TAIL_CAVEAT
            preselected = False
        elif resolved.confidence < 0.7:
            caveat = _LOW_CONFIDENCE_CAVEAT
            preselected = False
        else:
            caveat = None
            preselected = True

        ingredients.append(
            {
                "name": name,
                "food_id": resolved.food_id,
                "preselected": preselected,
                "confidence": resolved.confidence,
                "portion": portion,
                # ADR-0003 §6d ingredient-provenance vocab is curated|ai|user;
                # machine-generated rows are labeled 'ai' on both paths.
                "source": "ai",
                "thumbnail_id": resolved.thumbnail_id,
                "caveat": caveat,
            }
        )
    return ingredients, top_source


async def build_breakdown(
    db: AsyncSession,
    input_text: str,
    surface: str = "search",
    food_id: str | None = None,
) -> dict[str, Any]:
    """Produce the unified ADR-0003 §1 breakdown for an input.

    Precedence: curated → AI → heuristic (ADR-0003 §2). Every returned
    ingredient has passed through the canonical resolution layer (§3).
    """
    input_text = (input_text or "").strip()
    if not input_text:
        return {
            "input": "",
            "resolved_food_id": None,
            "source": "heuristic",
            "ingredients": [],
        }

    resolved_food_id = await _resolve_composite(db, input_text, food_id)

    # 1. Curated path — deterministic edges for a known composite.
    if resolved_food_id:
        curated = await _curated_ingredients(db, resolved_food_id)
        if curated:
            return {
                "input": input_text,
                "resolved_food_id": resolved_food_id,
                "source": "curated",
                "ingredients": curated,
            }

    # 2/3. AI floor → heuristic fallback (curated miss).
    ingredients, source = await _ai_ingredients(db, input_text)

    # Honest "couldn't break this down": a single bare token that did not split
    # and did not resolve is surfaced with a marker caveat rather than logged as
    # a blob (ADR-0003 §2 heuristic clause).
    if (
        source == "heuristic"
        and len(ingredients) == 1
        and ingredients[0]["food_id"] is None
        and ingredients[0]["name"].strip().lower() == input_text.lower()
    ):
        ingredients[0]["caveat"] = _NO_SPLIT_CAVEAT
        ingredients[0]["preselected"] = False

    return {
        "input": input_text,
        "resolved_food_id": resolved_food_id,
        "source": source,
        "ingredients": ingredients,
    }
