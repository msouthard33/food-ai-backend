"""User-scoped endpoints — profile actions, onboarding hooks."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.services import trigger_service, user_service

router = APIRouter(prefix="/api/v1/users", tags=["users"])

# Accepted condition strings (must match CONDITION_PRIORS keys)
VALID_CONDITIONS = {"ibs", "mcas", "histamine_intolerance", "food_allergy"}


class SeedPriorsRequest(BaseModel):
    conditions: list[str]

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, v: list[str]) -> list[str]:
        normalised = [c.lower().strip() for c in v]
        unknown = set(normalised) - VALID_CONDITIONS
        if unknown:
            raise ValueError(
                f"Unknown condition(s): {sorted(unknown)}. "
                f"Valid values: {sorted(VALID_CONDITIONS)}"
            )
        if not normalised:
            raise ValueError("conditions list must not be empty")
        return normalised


class SeedPriorsResponse(BaseModel):
    seeded: int
    component_types: list[str]


@router.post(
    "/me/seed-priors",
    response_model=SeedPriorsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Seed condition-based trigger priors for onboarding",
)
async def seed_priors(
    body: SeedPriorsRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SeedPriorsResponse:
    """Seed initial TriggerPrediction rows based on the user's declared conditions.

    Called once at onboarding so the trigger engine has a cold-start prior before
    the user has logged enough meals and symptoms for statistical inference.

    Idempotent: re-calling with the same conditions returns seeded=0 (no duplicates).

    Body:
        conditions: list of condition strings — one or more of:
            "ibs", "mcas", "histamine_intolerance", "food_allergy"

    Returns:
        seeded:          number of new TriggerPrediction rows created
        component_types: component type strings of the created rows
    """
    try:
        created = await trigger_service.seed_condition_priors(
            db=db,
            user_id=user.id,
            condition_types=body.conditions,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to seed condition priors: {exc}",
        ) from exc

    return SeedPriorsResponse(
        seeded=len(created),
        component_types=[str(p.component_type) for p in created],
    )


@router.delete(
    "/me",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Permanently delete the authenticated user's account and all data",
)
async def delete_my_account(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Irreversibly delete the account and all associated health data.

    Fulfils the GDPR Art. 17 / CCPA right to erasure. Removes the user row and every
    dependent PHI record via database ON DELETE CASCADE, best-effort deletes meal
    photos from Supabase Storage, and removes the Supabase Auth identity so the
    account cannot be re-authenticated. This action cannot be undone.
    """
    try:
        await user_service.delete_user_account(db, user)
    except Exception as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account",
        ) from exc
