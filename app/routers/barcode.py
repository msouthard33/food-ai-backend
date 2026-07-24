"""Barcode lookup endpoints.

Stub router — full Open Food Facts integration is planned for W2-3 capture sprint.
"""

from fastapi import APIRouter, Depends

from app.dependencies import get_current_user
from app.models.user import User

router = APIRouter(prefix="/api/v1/barcode", tags=["barcode"])


@router.get(
    "/{barcode}",
    response_model=dict,
    summary="Look up a product by barcode",
)
async def lookup_barcode(
    barcode: str,
    user: User = Depends(get_current_user),
) -> dict:
    """Look up a product by barcode number.

    Currently returns a stub response. Full Open Food Facts integration
    with KB matching and AI fallback is planned for a future sprint.
    """
    return {
        "barcode": barcode,
        "status": "not_implemented",
        "message": "Barcode lookup is not yet available. Full Open Food Facts integration coming soon.",
    }
