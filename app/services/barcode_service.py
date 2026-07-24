"""Barcode -> Open Food Facts -> KB trigger-profile pipeline (W2-3, Pillar 2).

Scan a barcode, look the product up in Open Food Facts (free, no license cost),
map the returned product name + ingredient list onto the food KB using the same
KB matcher the photo pipeline uses (``AIOrchestrator._search_kb_for_food``), and
return a structured trigger profile. If OFF has no product, or no KB match is
found, fall back to the AI decomposition path over the product name/ingredients
(the offline heuristic splitter reused from ``meal_decomposition``).

Design notes:
- ``fetch_off_product`` is the single seam that talks to the live OFF API. Tests
  patch it so the suite never touches the network.
- Every response carries a ``tier_label`` (D9 confidence vocabulary) at the
  product level AND on every ingredient. We NEVER return a bare numeric percent
  as the only confidence signal.
- No DB session required: the KB matcher reads the bundled KB JSON, mirroring the
  photo pipeline so barcode works on a fresh/DB-less deployment.
"""

from __future__ import annotations

import logging
import re

import httpx

from app.services.ai_orchestrator import AIOrchestrator
from app.services.llm_provider import confidence_to_tier
from app.services.meal_decomposition import _heuristic_split

logger = logging.getLogger(__name__)

OFF_BASE_URL = "https://world.openfoodfacts.org/api/v2/product"
OFF_TIMEOUT_SECONDS = 8.0
# OFF asks API consumers to identify themselves via a descriptive User-Agent.
OFF_USER_AGENT = "FoodAI/1.0 (dietary trigger tracker; contact: support@foodai.app)"

# Confidence floors for the AI-fallback / not-found paths.
_OFF_ONLY_CONFIDENCE = 0.30  # OFF has the product but the KB does not
_NOT_FOUND_CONFIDENCE = 0.10  # OFF does not know this barcode at all

_PAREN_RE = re.compile(r"\([^)]*\)")
_PCT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%")
# Ingredient tokens shorter than this (after cleaning) are dropped as noise.
_MIN_TOKEN_LEN = 3
_MAX_INGREDIENTS = 25


async def fetch_off_product(barcode: str) -> dict | None:
    """Fetch a product from Open Food Facts. Returns the product dict or None.

    Returns None when OFF does not recognise the barcode (``status`` != 1) or the
    product payload is empty. Network/HTTP errors propagate to the caller, which
    is responsible for degrading gracefully.

    This function is the ONLY place that performs live network I/O; tests patch it.
    """
    url = f"{OFF_BASE_URL}/{barcode}.json"
    async with httpx.AsyncClient(timeout=OFF_TIMEOUT_SECONDS) as client:
        resp = await client.get(url, headers={"User-Agent": OFF_USER_AGENT})
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") == 1 and data.get("product"):
        return data["product"]
    return None


def _clean_str(value: object) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    return text or None


def _clean_ingredient_token(raw: str) -> str | None:
    """Normalise a raw ingredient string into a matchable food token."""
    token = raw.lower()
    token = _PAREN_RE.sub(" ", token)
    token = _PCT_RE.sub(" ", token)
    token = token.replace("*", " ").replace("_", " ")
    token = re.sub(r"[^a-z0-9 &-]", " ", token)
    token = " ".join(token.split()).strip(" -&")
    if len(token) < _MIN_TOKEN_LEN or token.isdigit():
        return None
    return token


def _extract_ingredient_names(product: dict) -> list[str]:
    """Pull a de-duplicated list of ingredient tokens from an OFF product."""
    names: list[str] = []
    seen: set[str] = set()

    structured = product.get("ingredients")
    if isinstance(structured, list) and structured:
        raw_names = [i.get("text", "") for i in structured if isinstance(i, dict)]
    else:
        text = product.get("ingredients_text") or ""
        raw_names = re.split(r"[,;]", text)

    for raw in raw_names:
        cleaned = _clean_ingredient_token(str(raw))
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            names.append(cleaned)
        if len(names) >= _MAX_INGREDIENTS:
            break
    return names


async def _match_name(orchestrator: AIOrchestrator, name: str) -> dict | None:
    """Match a single food/ingredient name to the KB. Returns an ingredient profile or None."""
    from app.services.ai_orchestrator import extract_preparation_method

    base_name, prep_method = extract_preparation_method(name)
    try:
        allergen_summary, kb_name, kb_score = await orchestrator._search_kb_for_food(
            base_name, preparation_method=prep_method
        )
    except Exception as exc:  # noqa: BLE001 — a bad KB row must not kill the whole scan
        logger.warning("KB match failed for '%s': %s", name, str(exc)[:200])
        return None

    if kb_name is None or kb_score is None:
        return None

    confidence = round(float(kb_score), 3)
    return {
        "name": name,
        "kb_match_name": kb_name,
        "kb_match_score": confidence,
        "confidence": confidence,
        "tier_label": confidence_to_tier(confidence),
        "allergen_summary": allergen_summary,
    }


def _unmatched_ingredient(name: str) -> dict:
    return {
        "name": name,
        "kb_match_name": None,
        "kb_match_score": None,
        "confidence": 0.0,
        "tier_label": confidence_to_tier(0.0),
        "allergen_summary": None,
    }


async def lookup_barcode_profile(barcode: str) -> dict:
    """Full barcode -> trigger-profile pipeline.

    Returns a dict shaped for ``BarcodeProfileResponse`` in the router. Always
    carries a product-level ``tier_label`` plus a ``tier_label`` on every
    ingredient. Never raises for the ordinary "not found" case.
    """
    off_product: dict | None = None
    off_error: str | None = None
    try:
        off_product = await fetch_off_product(barcode)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully on any OFF failure
        off_error = type(exc).__name__
        logger.warning("OFF lookup failed for %s: %s", barcode, off_error)

    orchestrator = AIOrchestrator()

    # -- Case 1: OFF does not know this barcode ------------------------------
    if off_product is None:
        confidence = _NOT_FOUND_CONFIDENCE
        message = (
            "Product not found in Open Food Facts."
            if off_error is None
            else "Open Food Facts lookup is temporarily unavailable."
        )
        return {
            "barcode": barcode,
            "status": "not_found",
            "source": "openfoodfacts",
            "off_found": False,
            "product_name": None,
            "brands": None,
            "confidence": confidence,
            "tier_label": confidence_to_tier(confidence),
            "ingredients": [],
            "matched_count": 0,
            "message": message,
        }

    product_name = _clean_str(off_product.get("product_name"))
    brands = _clean_str(off_product.get("brands"))
    ingredient_names = _extract_ingredient_names(off_product)

    # -- Match the product name itself, then each listed ingredient ----------
    ingredient_profiles: list[dict] = []
    matched_scores: list[float] = []

    product_profile = None
    if product_name:
        product_profile = await _match_name(orchestrator, product_name)
        if product_profile:
            matched_scores.append(product_profile["confidence"])

    for name in ingredient_names:
        profile = await _match_name(orchestrator, name)
        if profile:
            ingredient_profiles.append(profile)
            matched_scores.append(profile["confidence"])
        else:
            ingredient_profiles.append(_unmatched_ingredient(name))

    # -- AI fallback: no KB matches yet, decompose the product name ----------
    source = "openfoodfacts+kb"
    if not matched_scores and product_name:
        source = "openfoodfacts+ai_decomposition"
        seen = {p["name"] for p in ingredient_profiles}
        for token, _portion in _heuristic_split(product_name):
            cleaned = _clean_ingredient_token(token)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            profile = await _match_name(orchestrator, cleaned)
            if profile:
                ingredient_profiles.append(profile)
                matched_scores.append(profile["confidence"])

    # -- Aggregate the product-level confidence ------------------------------
    if matched_scores:
        # The strongest single signal drives the tier; per-ingredient tiers
        # preserve the nuance for the UI.
        confidence = round(max(matched_scores), 3)
        status = "matched"
    else:
        confidence = _OFF_ONLY_CONFIDENCE
        status = "off_only"
        source = "openfoodfacts+ai_decomposition"

    return {
        "barcode": barcode,
        "status": status,
        "source": source,
        "off_found": True,
        "product_name": product_name,
        "brands": brands,
        "confidence": confidence,
        "tier_label": confidence_to_tier(confidence),
        "ingredients": ingredient_profiles,
        "matched_count": len([p for p in ingredient_profiles if p["kb_match_name"]]),
        "message": None,
    }
