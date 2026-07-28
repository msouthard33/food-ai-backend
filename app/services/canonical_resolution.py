"""Canonical ingredient resolution / alias-normalization layer — ADR-0003 §3.

This is the **non-negotiable substrate** of the hybrid breakdown architecture:
whichever path (curated / AI / heuristic) produces an ingredient checklist,
every ingredient string MUST collapse onto a single canonical KB ``food_id``
before it reaches the trigger engine — otherwise correlations fragment across
synonyms ("soy sauce" vs "shoyu" vs "tamari") and the known collinearity
problem gets *worse* rather than better.

Two stages:

1. **Normalize + alias-collapse (pure, offline, deterministic).**
   ``canonicalize(name)`` lowercases, strips punctuation/portions, collapses
   whitespace, and folds a curated synonym table onto one canonical term.
   This is a pure function — unit-tested without a DB — so the alias collapse
   is verifiable in isolation.

2. **Resolve to a canonical KB ``food_id``.** ``resolve_ingredient`` runs the
   canonical term through the existing pgvector semantic search
   (:mod:`app.services.semantic_search`) and returns the top match's
   ``food_id`` when the score clears ``RESOLVE_THRESHOLD``; otherwise it
   returns ``food_id=None`` with a "not yet in database" marker. Long-tail /
   ``None`` ingredients are candidates for promotion (ADR-0003 §5) and are
   surfaced honestly downstream (they raise ``coverage.components_unknown`` in
   Trigger Preview).

Non-goals (ADR-0003): this layer does not score components, does not decompose
composites (that is the breakdown service), and does not mint thumbnails.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services import semantic_search

# A match at or above this cosine-similarity score is trusted as a canonical
# resolution. Below it, we return food_id=None (long-tail) rather than binding
# the ingredient to a weak/wrong node — a wrong node is worse than an honest
# "not yet in database".
RESOLVE_THRESHOLD = 0.5

LONG_TAIL_MARKER = "not_yet_in_database"

# ---------------------------------------------------------------------------
# Alias table — curated synonym → canonical KB term.
#
# Keys are already-normalized (lowercase, singular-ish) alias strings; values
# are the canonical term we hand to semantic search. This is deliberately small
# and hand-curated (dietitian-vettable), not an ML guess. Grows over time; the
# promotion loop (ADR-0003 §5) is the mechanism that widens it.
# ---------------------------------------------------------------------------
_ALIAS_MAP: dict[str, str] = {
    # soy-sauce family — the canonical collinearity example from ADR-0003 §3
    "shoyu": "soy sauce",
    "tamari": "soy sauce",
    "shoyu sauce": "soy sauce",
    # alliums
    "scallion": "green onion",
    "scallions": "green onion",
    "spring onion": "green onion",
    "spring onions": "green onion",
    # regional produce names
    "aubergine": "eggplant",
    "courgette": "zucchini",
    "capsicum": "bell pepper",
    "rocket": "arugula",
    "coriander leaf": "cilantro",
    "garbanzo": "chickpeas",
    "garbanzo bean": "chickpeas",
    "garbanzo beans": "chickpeas",
    "chick pea": "chickpeas",
    "chick peas": "chickpeas",
    # proteins / seafood
    "prawn": "shrimp",
    "prawns": "shrimp",
    # dairy
    "curd": "yogurt",
    "dahi": "yogurt",
    # grains
    "beancurd": "tofu",
    "bean curd": "tofu",
}

# Light plural stripping — only the trivial trailing-"s"/"es" cases. We do NOT
# attempt real lemmatization (that risks mangling "molasses" → "molasse"), so
# irregulars are handled by the alias map above instead. "es" is only stripped
# for sibilant/"oes" endings (tomatoes→tomato, dishes→dish); everything else
# drops a bare trailing "s" (apples→apple, courgettes→courgette).
_SIBILANT_ES = ("ses", "xes", "zes", "ches", "shes", "oes")
_PLURAL_BLOCKLIST = {
    "molasses",
    "hummus",
    "couscous",
    "asparagus",
    "watercress",
    "greens",
    "oats",
    "grits",
}

_PORTION_PREFIX = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*"
    r"(?:cup|cups|tbsp|tsp|oz|g|grams?|ounces?|slices?|pieces?|cloves?|"
    r"handful|handfuls|serving|servings)?\s+",
    flags=re.IGNORECASE,
)
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def _depluralize(term: str) -> str:
    """Collapse trivial trailing plurals, guarding known mass/irregular nouns."""
    if term in _PLURAL_BLOCKLIST or len(term) < 4:
        return term
    if term.endswith("ss") or term.endswith("us"):
        return term  # mass/irregular ("watercress", "hummus", "citrus")
    if term.endswith(_SIBILANT_ES):
        return term[:-2]  # tomatoes -> tomato, dishes -> dish
    if term.endswith("s"):
        return term[:-1]  # apples -> apple, courgettes -> courgette
    return term


def canonicalize(name: str) -> str:
    """Normalize + alias-collapse an ingredient string to a canonical term.

    Pure and deterministic (no DB). Steps: lowercase → strip a leading portion
    phrase → drop punctuation → collapse whitespace → alias-map (full-string,
    then depluralized) → depluralize the residual.

    Examples::

        canonicalize("Shoyu")        -> "soy sauce"
        canonicalize("2 tbsp Tamari") -> "soy sauce"
        canonicalize("SOY SAUCE")    -> "soy sauce"
        canonicalize("Scallions")    -> "green onion"
        canonicalize("Tomatoes")     -> "tomato"
    """
    if not name:
        return ""
    term = name.strip().lower()
    term = _PORTION_PREFIX.sub("", term)
    term = _PUNCT.sub(" ", term)
    term = _WS.sub(" ", term).strip()
    if not term:
        return ""
    # Full-string alias hit first (handles multi-word aliases like "spring onion").
    if term in _ALIAS_MAP:
        return _ALIAS_MAP[term]
    # Depluralized alias hit (e.g. "prawns" -> "prawn" -> "shrimp").
    deplural = _depluralize(term)
    if deplural in _ALIAS_MAP:
        return _ALIAS_MAP[deplural]
    return deplural


# Coarse category placeholder classifier for thumbnail fallback (ADR-0003 §6):
# long-tail ingredients with no food_id get a generic category glyph key rather
# than a real per-food thumbnail. Keyword-based and intentionally conservative.
_PLACEHOLDER_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(sauce|dressing|mayo|aioli|ketchup|mustard|salsa|glaze|syrup)\b"), "sauce"),
    (re.compile(r"\b(rice|bread|noodle|pasta|oat|wheat|grain|tortilla|flour|cereal)\b"), "grain"),
    (re.compile(r"\b(chicken|beef|pork|fish|tofu|egg|shrimp|turkey|lamb|bean|lentil)\b"), "protein"),
    (re.compile(r"\b(milk|cheese|yogurt|cream|butter)\b"), "dairy"),
    (re.compile(r"\b(apple|berry|banana|grape|orange|mango|melon|fruit)\b"), "fruit"),
    (re.compile(r"\b(lettuce|spinach|kale|broccoli|carrot|pepper|onion|vegetable)\b"), "vegetable"),
]


def placeholder_key(canonical_term: str) -> str:
    """Return a category-placeholder thumbnail key for a long-tail ingredient.

    No image pipeline runs here (ADR-0003 §6 / build step 6 is deferred); this
    only picks the *key* the client would map to a bundled category glyph.
    """
    for pattern, cat in _PLACEHOLDER_RULES:
        if pattern.search(canonical_term):
            return f"placeholder:{cat}"
    return "placeholder:generic"


@dataclass
class ResolvedIngredient:
    input: str
    canonical_term: str
    food_id: str | None  # canonical KB food_database.id (str uuid) or None (long-tail)
    matched_name: str | None
    confidence: float  # 0..1 semantic score (0 when unresolved)
    resolved: bool
    marker: str | None  # LONG_TAIL_MARKER when food_id is None
    thumbnail_id: str  # food_id (str) when resolved, else a placeholder key

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def resolve_ingredient(
    db: AsyncSession,
    name: str,
    provider: str = "offline",
) -> ResolvedIngredient:
    """Resolve one ingredient string to a canonical KB ``food_id`` (or None).

    Applies :func:`canonicalize` (alias collapse) then semantic search. A match
    at/above :data:`RESOLVE_THRESHOLD` binds the canonical ``food_id``; below it
    (or no match) the ingredient is long-tail: ``food_id=None`` +
    :data:`LONG_TAIL_MARKER`. This is the single choke point every downstream
    ingredient passes through (ADR-0003 §3).
    """
    canonical = canonicalize(name)
    if not canonical:
        return ResolvedIngredient(
            input=name,
            canonical_term="",
            food_id=None,
            matched_name=None,
            confidence=0.0,
            resolved=False,
            marker=LONG_TAIL_MARKER,
            thumbnail_id="placeholder:generic",
        )

    matches = await semantic_search.semantic_search(db, canonical, limit=1, provider=provider)
    if matches and matches[0].score >= RESOLVE_THRESHOLD:
        top = matches[0]
        fid = str(top.food_id)
        return ResolvedIngredient(
            input=name,
            canonical_term=canonical,
            food_id=fid,
            matched_name=top.name,
            confidence=round(float(top.score), 3),
            resolved=True,
            marker=None,
            thumbnail_id=fid,
        )

    return ResolvedIngredient(
        input=name,
        canonical_term=canonical,
        food_id=None,
        matched_name=None,
        confidence=round(float(matches[0].score), 3) if matches else 0.0,
        resolved=False,
        marker=LONG_TAIL_MARKER,
        thumbnail_id=placeholder_key(canonical),
    )
