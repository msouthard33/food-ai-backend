"""User-scoped endpoints — profile actions, onboarding hooks."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.enums import ComponentType, Severity
from app.models.user import User, UserKnownAllergen
from app.services import trigger_service, user_service

router = APIRouter(prefix="/api/v1/users", tags=["users"])

# Accepted condition strings (must match CONDITION_PRIORS keys)
VALID_CONDITIONS = {"ibs", "mcas", "histamine_intolerance", "food_allergy"}

# Accepted allergen / sensitivity identifiers — the full ComponentType domain,
# which is exactly the column domain of user_known_allergens.allergen_type.
VALID_ALLERGENS = {e.value for e in ComponentType}
VALID_SEVERITIES = {e.value for e in Severity}


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


# ---------------------------------------------------------------------------
# Allergen / sensitivity persistence (BLK-MOBILE-ALLERGEN-PERSIST)
#
# Onboarding captures the user's allergen multi-select. Conditions already
# persist via /me/seed-priors; this endpoint gives the client somewhere to
# write the allergen/sensitivity selections. Persistence targets the existing
# user_known_allergens table (allergen_type column is a ComponentType), so no
# schema change is required.
#
# Semantics: full-replace (PUT). The submitted set becomes the user's complete
# known-allergen set. Idempotent — re-submitting the same set yields the same
# state and returns 200 without creating duplicate rows (guarded by the
# uq (user_id, allergen_type) constraint and an explicit reconcile).
# ---------------------------------------------------------------------------


class AllergenSelection(BaseModel):
    """A single allergen/sensitivity selection.

    The mobile onboarding multi-select sends bare identifier strings; this model
    also accepts optional per-allergen metadata for richer clients.
    """

    allergen: str
    confirmed: bool = False
    severity: str | None = None
    reaction_notes: str | None = None

    @field_validator("allergen")
    @classmethod
    def validate_allergen(cls, v: str) -> str:
        normalised = v.lower().strip()
        if normalised not in VALID_ALLERGENS:
            raise ValueError(
                f"Unknown allergen: {v!r}. Valid values: {sorted(VALID_ALLERGENS)}"
            )
        return normalised

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str | None) -> str | None:
        if v is None:
            return None
        normalised = v.lower().strip()
        if normalised not in VALID_SEVERITIES:
            raise ValueError(
                f"Unknown severity: {v!r}. Valid values: {sorted(VALID_SEVERITIES)}"
            )
        return normalised


class SetAllergensRequest(BaseModel):
    """Full-replace payload for the user's known-allergen set.

    ``allergens`` accepts either bare identifier strings (the onboarding
    multi-select shape) or objects with optional metadata. An empty list is
    valid and clears the user's allergen set.
    """

    allergens: list[str | AllergenSelection]

    @field_validator("allergens")
    @classmethod
    def coerce_and_validate(cls, v: list) -> list[AllergenSelection]:
        selections: list[AllergenSelection] = []
        for item in v:
            if isinstance(item, str):
                selections.append(AllergenSelection(allergen=item))
            elif isinstance(item, AllergenSelection):
                selections.append(item)
            else:  # pragma: no cover — pydantic coerces dicts before this runs
                selections.append(AllergenSelection(**item))
        # Reject duplicate identifiers within a single request.
        seen = [s.allergen for s in selections]
        dupes = sorted({a for a in seen if seen.count(a) > 1})
        if dupes:
            raise ValueError(f"Duplicate allergen(s) in request: {dupes}")
        return selections


class AllergenOut(BaseModel):
    allergen: str
    confirmed: bool
    severity: str | None = None
    reaction_notes: str | None = None


class AllergenProfileResponse(BaseModel):
    allergens: list[AllergenOut]
    count: int


def _serialise_allergens(rows: list[UserKnownAllergen]) -> AllergenProfileResponse:
    ordered = sorted(rows, key=lambda r: str(r.allergen_type))
    out = [
        AllergenOut(
            allergen=str(r.allergen_type.value if hasattr(r.allergen_type, "value") else r.allergen_type),
            confirmed=r.confirmed,
            severity=(r.severity.value if r.severity is not None else None),
            reaction_notes=r.reaction_notes,
        )
        for r in ordered
    ]
    return AllergenProfileResponse(allergens=out, count=len(out))


@router.put(
    "/me/allergens",
    response_model=AllergenProfileResponse,
    summary="Replace the authenticated user's known-allergen set",
)
async def set_my_allergens(
    body: SetAllergensRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AllergenProfileResponse:
    """Persist the user's allergen/sensitivity selections (onboarding + settings).

    Full-replace semantics: the submitted set becomes the user's complete
    known-allergen set. Idempotent — re-submitting the same selections leaves
    the state unchanged and returns 200. Rows for allergens no longer selected
    are removed; existing rows for still-selected allergens are updated in place
    (preserving no stale duplicates thanks to uq(user_id, allergen_type)).

    Body:
        allergens: list of allergen identifiers (ComponentType values, e.g.
            "peanuts", "tree_nuts", "milk_dairy", "shellfish", "gluten"), or
            objects {allergen, confirmed?, severity?, reaction_notes?}.

    Returns the user's full known-allergen set after the update.
    """
    selections: list[AllergenSelection] = body.allergens  # coerced by validator
    target = {s.allergen: s for s in selections}

    result = await db.execute(
        select(UserKnownAllergen).where(UserKnownAllergen.user_id == user.id)
    )
    existing = list(result.scalars().all())
    existing_by_type = {
        str(r.allergen_type.value if hasattr(r.allergen_type, "value") else r.allergen_type): r
        for r in existing
    }

    # Remove allergens no longer selected.
    for allergen_value, row in existing_by_type.items():
        if allergen_value not in target:
            await db.delete(row)

    # Upsert selected allergens.
    for allergen_value, sel in target.items():
        severity = Severity(sel.severity) if sel.severity is not None else None
        if allergen_value in existing_by_type:
            row = existing_by_type[allergen_value]
            row.confirmed = sel.confirmed
            row.severity = severity
            row.reaction_notes = sel.reaction_notes
        else:
            db.add(
                UserKnownAllergen(
                    user_id=user.id,
                    allergen_type=ComponentType(allergen_value),
                    confirmed=sel.confirmed,
                    severity=severity,
                    reaction_notes=sel.reaction_notes,
                )
            )

    await db.flush()

    refreshed = await db.execute(
        select(UserKnownAllergen).where(UserKnownAllergen.user_id == user.id)
    )
    rows = list(refreshed.scalars().all())
    await db.commit()
    return _serialise_allergens(rows)


@router.get(
    "/me/allergens",
    response_model=AllergenProfileResponse,
    summary="Get the authenticated user's known-allergen set",
)
async def get_my_allergens(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AllergenProfileResponse:
    """Return the user's persisted allergen/sensitivity selections."""
    result = await db.execute(
        select(UserKnownAllergen).where(UserKnownAllergen.user_id == user.id)
    )
    return _serialise_allergens(list(result.scalars().all()))


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
