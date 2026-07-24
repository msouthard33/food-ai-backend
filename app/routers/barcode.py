"""Barcode lookup endpoints (W2-3, Pillar 2 — Capture Without Friction).

Scan a barcode -> Open Food Facts -> KB trigger profile. Every response carries
a D9 ``tier_label`` at the product level and on each ingredient — never a bare
numeric percent as the only confidence signal.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user
from app.models.user import User
from app.services import barcode_service

router = APIRouter(prefix="/api/v1/barcode", tags=["barcode"])


class IngredientProfile(BaseModel):
    name: str
    kb_match_name: str | None = None
    kb_match_score: float | None = None
    confidence: float
    tier_label: str
    allergen_summary: dict | None = None


class BarcodeProfileResponse(BaseModel):
    barcode: str
    # "matched" | "off_only" | "not_found"
    status: str
    # "openfoodfacts+kb" | "openfoodfacts+ai_decomposition" | "openfoodfacts"
    source: str
    off_found: bool
    product_name: str | None = None
    brands: str | None = None
    confidence: float
    tier_label: str
    ingredients: list[IngredientProfile]
    matched_count: int
    message: str | None = None


@router.get(
    "/{barcode}",
    response_model=BarcodeProfileResponse,
    summary="Look up a product by barcode and return its trigger profile",
    description=(
        "Scans a barcode against Open Food Facts, maps the product and its "
        "ingredients onto the food knowledge base, and returns a structured "
        "trigger profile. Falls back to AI decomposition of the product name "
        "when no direct KB match is found. Every response includes a confidence "
        "tier label (Well-established / Some evidence / AI estimate)."
    ),
)
async def lookup_barcode(
    barcode: str,
    user: User = Depends(get_current_user),
) -> BarcodeProfileResponse:
    """Look up a product by barcode number and return a structured trigger profile."""
    result = await barcode_service.lookup_barcode_profile(barcode)
    return BarcodeProfileResponse(**result)
