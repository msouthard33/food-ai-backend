"""Food database search endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.food import FoodSearchListOut, FoodSearchResult
from app.services import (
    breakdown_service,
    food_service,
    meal_decomposition,
    semantic_search,
)

router = APIRouter(prefix="/api/v1/foods", tags=["foods"])


@router.get(
    "/search",
    response_model=FoodSearchListOut,
    summary="Search the food knowledge base",
)
async def search_foods(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    category: str | None = Query(None, description="Filter by food category"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FoodSearchListOut:
    foods, total = await food_service.search_foods(db, q, category=category, limit=limit)
    return FoodSearchListOut(
        total=total,
        query=q,
        items=[FoodSearchResult.model_validate(f) for f in foods],
    )


@router.get(
    "/search/semantic",
    summary="pgvector semantic search — W2-1 Pillar 1",
)
async def search_foods_semantic(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(10, ge=1, le=50),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Semantic food search over pgvector embeddings.

    Falls back to lexical search when the embedding table is empty, so this
    never dead-ends on a fresh DB (embeddings are populated via a gated
    follow-up).
    """
    matches = await semantic_search.semantic_search(db, q, limit=limit)
    return {
        "query": q,
        "total": len(matches),
        "items": [
            {
                "food_id": str(m.food_id),
                "name": m.name,
                "score": m.score,
                "source": m.source,
            }
            for m in matches
        ],
    }


@router.post(
    "/decompose",
    summary="Compositional meal decomposition — W2-1 Pillar 1",
)
async def decompose_meal(
    payload: dict,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Decompose free-text meal (`{"meal_text": "..."}`) into structured ingredients.

    Uses the LLM decomposition path when available, otherwise a deterministic
    rule-based splitter; each ingredient is resolved against the KB via semantic
    search.
    """
    meal_text = str(payload.get("meal_text", "")).strip()
    return await meal_decomposition.decompose_meal_text(db, meal_text)


@router.post(
    "/breakdown",
    summary="Unified ingredient breakdown — ADR-0003 §1 (curated→AI→heuristic)",
)
async def breakdown_food(
    payload: dict,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the unified ADR-0003 §1 breakdown for an input.

    Body: ``{"input": str, "surface": "search"|"capture", "food_id": <uuid|null>}``.

    Resolution precedence is curated → AI → heuristic (ADR-0003 §2). Every
    ingredient is re-resolved through the canonical alias/normalization layer
    (§3) so it lands on a canonical KB ``food_id`` or an honest ``null``
    long-tail marker. The response shape matches the Trigger Preview §4a
    contract verbatim, so Trigger Preview drops in as a consumer.

    JWT-required, consistent with ``/foods/search``. ``/foods/decompose`` is
    retained unchanged for existing callers.
    """
    input_text = str(payload.get("input", payload.get("meal_text", ""))).strip()
    surface = str(payload.get("surface", "search"))
    food_id = payload.get("food_id")
    food_id = str(food_id) if food_id else None
    return await breakdown_service.build_breakdown(
        db, input_text, surface=surface, food_id=food_id
    )
