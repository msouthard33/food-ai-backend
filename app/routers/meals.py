"""Meal logging endpoints."""

import base64
import binascii
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import PaginationParams, get_current_user
from app.models.user import User
from app.schemas.meal import MealCreate, MealItemBatchCreate, MealItemOut, MealListOut, MealOut
from app.services import meal_service
from app.services.ai_orchestrator import AIOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/meals", tags=["meals"])


# ---------------------------------------------------------------------------
# Schemas for photo analysis
# ---------------------------------------------------------------------------

# Upload limits / accepted formats for meal photos.
# ~10 MB decoded; base64 inflates by ~4/3, so cap the encoded string accordingly
# (+a small margin for an optional data-URI prefix and whitespace).
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_B64_LEN = (_MAX_IMAGE_BYTES * 4) // 3 + 1024


def _sniff_image_type(head: bytes) -> str | None:
    """Return a format label if `head` starts with a supported image's magic bytes."""
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    # HEIC/HEIF: `....ftyp` followed by a brand such as heic/heix/mif1/heif
    if head[4:8] == b"ftyp" and head[8:12] in (
        b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1", b"msf1", b"heif"
    ):
        return "heic"
    return None


class PhotoAnalyzeRequest(BaseModel):
    image_base64: str

    @field_validator("image_base64")
    @classmethod
    def validate_image(cls, v: str) -> str:
        if not v:
            raise ValueError("image_base64 must not be empty")
        if len(v) > _MAX_B64_LEN:
            raise ValueError(
                f"Image too large: exceeds {_MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
            )
        # Strip an optional data-URI prefix ("data:image/jpeg;base64,....").
        payload = v.split(",", 1)[1] if v.startswith("data:") else v
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image_base64 is not valid base64") from exc
        if len(raw) > _MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image too large: exceeds {_MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
            )
        if _sniff_image_type(raw[:16]) is None:
            raise ValueError(
                "Unsupported image format (expected JPEG, PNG, WEBP, or HEIC/HEIF)"
            )
        return v


class FoodItemResponse(BaseModel):
    name: str
    portion: str
    confidence: float
    tier_label: str
    allergen_summary: dict | None = None
    kb_match_name: str | None = None
    kb_match_score: float | None = None
    error: str | None = None


class PhotoAnalyzeResponse(BaseModel):
    foods: list[FoodItemResponse]
    processing_time_ms: int
    photo_analysis_model: str
    search_model: str
    food_count: int
    cost_estimate_usd: float | None = None


# ---------------------------------------------------------------------------
# Meal CRUD
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=MealOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new meal entry",
)
async def create_meal(
    data: MealCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MealOut:
    """Create a new meal log entry.

    Photo upload is handled via a separate PATCH /api/v1/meals/{meal_id}/photo endpoint (coming soon).
    """
    meal = await meal_service.create_meal(db, user.id, data)
    # Re-fetch with eager-loaded relationships to avoid lazy-load errors
    meal = await meal_service.get_meal(db, meal.id, user.id)
    return MealOut.model_validate(meal)


@router.get(
    "",
    response_model=MealListOut,
    summary="List user's meals with pagination",
)
async def list_meals(
    pagination: PaginationParams = Depends(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MealListOut:
    meals, total = await meal_service.list_meals(
        db, user.id, offset=pagination.offset, limit=pagination.page_size
    )
    return MealListOut(
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        items=[MealOut.model_validate(m) for m in meals],
    )


@router.get(
    "/{meal_id}",
    response_model=MealOut,
    summary="Get a single meal by ID",
)
async def get_meal(
    meal_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MealOut:
    """Fetch a single meal entry by ID. Only returns meals owned by the authenticated user."""
    meal = await meal_service.get_meal(db, meal_id, user.id)
    if not meal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meal not found")
    return MealOut.model_validate(meal)


@router.post(
    "/{meal_id}/items",
    response_model=list[MealItemOut],
    status_code=status.HTTP_201_CREATED,
    summary="Add parsed food items to a meal",
)
async def add_meal_items(
    meal_id: uuid.UUID,
    data: MealItemBatchCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[MealItemOut]:
    # Verify meal ownership
    meal = await meal_service.get_meal(db, meal_id, user.id)
    if not meal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meal not found")

    items = await meal_service.add_meal_items(db, meal_id, data.items)
    # Re-fetch to get components loaded
    meal = await meal_service.get_meal(db, meal_id, user.id)
    return [MealItemOut.model_validate(item) for item in (meal.items if meal else [])]


# ---------------------------------------------------------------------------
# Photo analysis (W2-3)
# ---------------------------------------------------------------------------

@router.post(
    "/analyze-photo",
    response_model=PhotoAnalyzeResponse,
    summary="Analyze a meal photo to identify foods and allergens",
    description=(
        "Accepts a base64-encoded photo, identifies foods via AI vision, "
        "and returns structured allergen/sensitivity profiles from the KB. "
        "Image bytes are never logged or stored to disk."
    ),
)
async def analyze_photo(
    data: PhotoAnalyzeRequest,
    user: User = Depends(get_current_user),
) -> PhotoAnalyzeResponse:
    """Analyze a meal photo and return identified foods with allergen profiles.

    Pipeline: vision model -> food identification -> KB search -> allergen summary.
    Requires authenticated user (Supabase JWT).
    """
    # Never log image bytes (patient health data)
    logger.info("Photo analysis request from user=%s, image_size=%d bytes",
                user.id, len(data.image_base64))

    orchestrator = AIOrchestrator()
    result = await orchestrator.process_meal_photo(data.image_base64)

    return PhotoAnalyzeResponse(
        foods=[
            FoodItemResponse(
                name=f.name,
                portion=f.portion,
                confidence=f.confidence,
                tier_label=f.tier_label,
                allergen_summary=f.allergen_summary,
                kb_match_name=f.kb_match_name,
                kb_match_score=f.kb_match_score,
                error=f.error,
            )
            for f in result.foods
        ],
        processing_time_ms=result.processing_time_ms,
        photo_analysis_model=result.photo_analysis_model,
        search_model=result.search_model,
        food_count=result.food_count,
        cost_estimate_usd=result.cost_estimate_usd,
    )
