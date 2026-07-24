"""Compositional meal decomposition — W2-1 Pillar 1.

Takes free-text meal entries ("chicken salad with grapes on sourdough") and
returns a structured list of ingredients, each resolved against the food KB
via semantic search, each carrying a confidence score.

Architecture:
1. LLM pass ("chicken salad" -> ["chicken breast", "mayonnaise", "celery",
   "grapes", "sourdough bread"]) using the existing ai_orchestrator
   food_decomposition template.
2. For each ingredient, pgvector semantic search against the KB.
3. Attach confidence = min(llm_confidence, semantic_score).
4. Attach a plain-English caveat when confidence < 0.7.

Offline mode: if no LLM API key is configured, fall back to a deterministic
rule-based splitter (split on common connectors: 'with', 'and', ',', 'on',
'topped with'). This lets the test harness run the full pipeline without
live API calls — benchmarks still measure structured-output rate and
semantic match rate end-to-end.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services import semantic_search
from app.services.llm_provider import LLMProvider, get_llm_provider

logger = logging.getLogger(__name__)

DECOMPOSITION_PROMPT_VERSION = "w2-1.0.0"

_CONNECTOR_RE = re.compile(
    r"\s*(?:,|\band\b|\bwith\b|\bon\b|\btopped with\b|\bplus\b|\balongside\b|\b\+\b)\s*",
    flags=re.IGNORECASE,
)
_PORTION_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(cup|cups|tbsp|tsp|oz|g|grams|ounce|ounces|slice|slices|piece|pieces)?\s+",
    flags=re.IGNORECASE,
)


@dataclass
class DecomposedIngredient:
    ingredient: str
    portion: str | None
    confidence: float  # 0..1
    kb_match_id: str | None  # UUID as string, or None if unresolved
    kb_match_name: str | None
    source: str  # "llm" | "heuristic"
    caveat: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _heuristic_split(meal_text: str) -> list[tuple[str, str | None]]:
    """Rule-based fallback splitter. Returns [(ingredient, portion)]."""
    parts = [p.strip() for p in _CONNECTOR_RE.split(meal_text) if p and p.strip()]
    out: list[tuple[str, str | None]] = []
    for p in parts:
        m = _PORTION_RE.match(p)
        if m:
            portion = m.group(0).strip()
            ingredient = p[m.end() :].strip()
        else:
            portion = None
            ingredient = p
        if ingredient:
            out.append((ingredient, portion))
    return out


async def decompose_meal_text(
    db: AsyncSession,
    meal_text: str,
    provider: LLMProvider | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Decompose a free-text meal into structured ingredients with KB matches.

    Returns a dict:
        {
          "meal_text": str,
          "ingredients": [DecomposedIngredient, ...],
          "structured": bool,       # True if we produced at least 1 structured row
          "source": "llm"|"heuristic",
          "prompt_version": str,
        }
    """
    meal_text = (meal_text or "").strip()
    if not meal_text:
        return {
            "meal_text": "",
            "ingredients": [],
            "structured": False,
            "source": "heuristic",
            "prompt_version": DECOMPOSITION_PROMPT_VERSION,
        }

    raw_items: list[tuple[str, str | None, float]] = []  # (ingredient, portion, llm_conf)
    source = "heuristic"

    if use_llm:
        try:
            provider = provider or get_llm_provider()
            system = (
                "You are a clinical food decomposition engine. Given a free-text meal "
                "description, return STRICT JSON of the form "
                '{"ingredients":[{"ingredient":str,"portion":str|null,"confidence":float}]}. '
                "Do not invent ingredients. Do not include cooking verbs. Lowercase."
            )
            resp = await provider.chat(system_prompt=system, user_message=meal_text, max_tokens=600)
            parsed = resp.parse_json()
            items = parsed.get("ingredients", [])
            for it in items:
                if not isinstance(it, dict):
                    continue
                ing = str(it.get("ingredient", "")).strip()
                if not ing:
                    continue
                raw_items.append(
                    (
                        ing,
                        it.get("portion"),
                        float(it.get("confidence", 0.8)),
                    )
                )
            if raw_items:
                source = "llm"
        except Exception as e:  # noqa: BLE001 — any LLM failure falls through to heuristic
            logger.info("LLM decomposition unavailable (%s); using heuristic", type(e).__name__)

    if not raw_items:
        for ing, portion in _heuristic_split(meal_text):
            raw_items.append((ing, portion, 0.55))
        source = "heuristic"

    # Resolve each ingredient against the KB via semantic search
    resolved: list[DecomposedIngredient] = []
    for ingredient, portion, llm_conf in raw_items:
        matches = await semantic_search.semantic_search(db, ingredient, limit=1)
        if matches:
            top = matches[0]
            conf = round(min(llm_conf, max(top.score, 0.0)), 3)
            caveat = (
                None
                if conf >= 0.7
                else "Best-guess match — confirm ingredient before relying on trigger data."
            )
            resolved.append(
                DecomposedIngredient(
                    ingredient=ingredient,
                    portion=portion,
                    confidence=conf,
                    kb_match_id=str(top.food_id),
                    kb_match_name=top.name,
                    source=source,
                    caveat=caveat,
                )
            )
        else:
            resolved.append(
                DecomposedIngredient(
                    ingredient=ingredient,
                    portion=portion,
                    confidence=round(llm_conf * 0.4, 3),
                    kb_match_id=None,
                    kb_match_name=None,
                    source=source,
                    caveat="No KB match — this ingredient is not yet in our database.",
                )
            )

    return {
        "meal_text": meal_text,
        "ingredients": [r.to_dict() for r in resolved],
        "structured": bool(resolved),
        "source": source,
        "prompt_version": DECOMPOSITION_PROMPT_VERSION,
    }
