"""Elimination protocol endpoints — Day-One Value (Pillar 5)."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.sensitivity import UserSensitivityProfile
from app.models.user import User
from app.schemas.insights import ProtocolStartOut, ProtocolStartRequest

router = APIRouter(prefix="/api/v1/protocols", tags=["insights"])

# Pre-populated foods-to-avoid lists per protocol type (from KB knowledge)
_PROTOCOL_FOODS: dict[str, list[str]] = {
    "low-histamine": [
        "Aged Cheese",
        "Red Wine",
        "Sauerkraut",
        "Smoked Salmon",
        "Canned Tuna",
        "Soy Sauce",
        "Fermented Foods",
        "Spinach",
        "Tomatoes",
        "Avocado",
        "Eggplant",
        "Vinegar",
        "Craft Beer IPA",
        "Kimchi",
        "Miso",
        "Kombucha",
        "Tempeh",
    ],
    "top8-allergen": [
        "Milk",
        "Eggs",
        "Peanuts",
        "Tree Nuts",
        "Wheat",
        "Soy",
        "Fish",
        "Shellfish",
        "Sesame",
    ],
    "low-fodmap": [
        "Garlic",
        "Onion",
        "Wheat",
        "Apples",
        "Pears",
        "Watermelon",
        "Honey",
        "Milk",
        "Yogurt",
        "Cauliflower",
        "Mushrooms",
        "Beans",
        "Lentils",
        "Chickpeas",
        "Cashews",
        "Pistachios",
    ],
}

VALID_PROTOCOLS = set(_PROTOCOL_FOODS.keys())


@router.post(
    "/start",
    response_model=ProtocolStartOut,
    status_code=status.HTTP_201_CREATED,
    summary="Start an elimination protocol",
)
async def start_protocol(
    data: ProtocolStartRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProtocolStartOut:
    """Start a pre-populated elimination protocol for the authenticated user.

    Creates a record in user_sensitivity_profiles and returns the
    foods-to-avoid list for the chosen protocol type.
    """
    if data.protocol_type not in VALID_PROTOCOLS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid protocol_type. Must be one of: {', '.join(sorted(VALID_PROTOCOLS))}",
        )

    now = datetime.now(timezone.utc)
    component = _protocol_to_component(data.protocol_type)

    # Check if user already has an active profile for this component type
    existing = await db.execute(
        select(UserSensitivityProfile).where(
            UserSensitivityProfile.user_id == user.id,
            UserSensitivityProfile.component_type == component,
        )
    )
    profile = existing.scalar_one_or_none()

    if profile:
        # Reactivate existing profile
        profile.active = True
        profile.notes = f"Elimination protocol: {data.protocol_type}"
        profile.updated_at = now
        protocol_id = profile.id
    else:
        protocol_id = uuid.uuid4()
        profile = UserSensitivityProfile(
            id=protocol_id,
            user_id=user.id,
            component_type=component,
            weight=1.00,
            threshold=5.0,
            active=True,
            notes=f"Elimination protocol: {data.protocol_type}",
            created_at=now,
            updated_at=now,
        )
        db.add(profile)

    await db.flush()

    foods_to_avoid = _PROTOCOL_FOODS.get(data.protocol_type, [])

    return ProtocolStartOut(
        protocol_id=protocol_id,
        protocol_type=data.protocol_type,
        started_at=now,
        foods_to_avoid=foods_to_avoid,
    )


def _protocol_to_component(protocol_type: str) -> str:
    """Map protocol type to a primary ComponentType for the profile record."""
    mapping = {
        "low-histamine": "histamines",
        "top8-allergen": "gluten",  # top-8 uses gluten as lead component
        "low-fodmap": "fodmap",
    }
    return mapping.get(protocol_type, "other")
