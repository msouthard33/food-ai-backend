"""Auth endpoints.

DEMO-ONLY SURFACE. This router exists so the mobile app can enter a fully
authenticated demo session WITHOUT a working Supabase project. It does NOT
implement real user sign-up / sign-in — production auth is handled by Supabase
and validated in ``app.dependencies.get_current_user``.

The demo-login endpoint only ever issues tokens for a small set of FIXED demo
personas (deterministic identities). It can never impersonate an arbitrary user.
Guard it in real production by setting ``DEMO_LOGIN_ENABLED=false``.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from jose import jwt
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.enums import ConditionType
from app.models.user import User, UserCondition
from app.services import trigger_service

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# Fixed namespace so demo user UUIDs are deterministic and stable across calls,
# processes, and machines — the demo-seed script and the iOS app derive the same
# ids from the same (namespace, persona) pair.
_DEMO_NAMESPACE = uuid.UUID("11111111-2222-4333-8444-555555555555")

# The demo personas. Each maps to the condition string(s) understood by
# trigger_service.seed_condition_priors / CONDITION_PRIORS.
Persona = Literal["ibs", "mcas", "histamine"]

_PERSONA_CONDITIONS: dict[str, list[str]] = {
    "ibs": ["ibs"],
    "mcas": ["mcas"],
    "histamine": ["histamine_intolerance"],
}


def demo_user_id(persona: str) -> uuid.UUID:
    """Deterministic demo user id for a persona (uuid5 of a fixed namespace)."""
    return uuid.uuid5(_DEMO_NAMESPACE, f"demo:{persona}")


def demo_user_email(persona: str) -> str:
    """Deterministic demo user email for a persona."""
    return f"demo_{persona}@foodai.demo"


class DemoLoginRequest(BaseModel):
    persona: Persona = "ibs"


class DemoLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: uuid.UUID
    persona: str
    expires_in: int


def _mint_access_token(user_id: uuid.UUID, email: str, ttl_seconds: int) -> str:
    """Mint a token accepted by ``get_current_user``.

    When ``SUPABASE_JWT_SECRET`` is configured (production / staging), self-mint an
    HS256 JWT carrying the exact claims Supabase would issue and that
    ``_decode_supabase_jwt`` validates: ``sub``, ``aud="authenticated"``,
    ``role="authenticated"``, ``email``, ``iat``, ``exp``.

    When no secret is configured (local dev), ``get_current_user`` falls back to
    accepting a raw UUID as the bearer token, so return the user id directly.
    """
    settings = get_settings()
    now = datetime.now(UTC)

    if not settings.supabase_jwt_secret:
        # Dev shortcut: get_current_user accepts a raw UUID when no secret is set.
        return str(user_id)

    claims = {
        "sub": str(user_id),
        "aud": "authenticated",
        "role": "authenticated",
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl_seconds)).timestamp()),
    }
    return jwt.encode(claims, settings.supabase_jwt_secret, algorithm="HS256")


@router.post(
    "/demo-login",
    response_model=DemoLoginResponse,
    summary="Enter a fully-authenticated demo session (demo-only, no Supabase)",
)
async def demo_login(
    body: DemoLoginRequest,
    db: AsyncSession = Depends(get_db),
) -> DemoLoginResponse:
    """Create-or-get a fixed demo persona and return an auth token for it.

    DEMO-ONLY. Lets the mobile app showcase day-one value without a working
    Supabase project. Behaviour:

    * The persona maps to a DETERMINISTIC identity — a stable uuid5 id and a
      ``demo_{persona}@foodai.demo`` email — so repeat calls return the same user
      and the demo-seed script can target the same rows.
    * On first login the demo user is provisioned with its condition(s) and its
      cold-start trigger priors are seeded (``seed_condition_priors``), so a fresh
      demo session already has day-one insights.
    * The returned token is accepted by ``get_current_user``: an HS256 JWT signed
      with ``SUPABASE_JWT_SECRET`` when configured, else the raw UUID dev token.

    Disable in real production with ``DEMO_LOGIN_ENABLED=false``.
    """
    settings = get_settings()
    if not settings.demo_login_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )

    persona = body.persona
    user_id = demo_user_id(persona)
    email = demo_user_email(persona)
    conditions = _PERSONA_CONDITIONS[persona]

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        # First login for this persona — provision the deterministic demo user,
        # attach its condition(s), and seed cold-start priors for day-one value.
        user = User(
            id=user_id,
            email=email,
            display_name=f"Demo ({persona.upper()})",
            onboarding_completed=True,
            is_synthetic=True,  # reuse is_synthetic to flag non-real accounts
        )
        db.add(user)
        for cond in conditions:
            db.add(
                UserCondition(
                    user_id=user_id,
                    condition_type=ConditionType(cond),
                    diagnosed_by_doctor=False,
                )
            )
        await db.flush()
        await db.commit()

        # Seed priors on its own path (commits internally, idempotent).
        await trigger_service.seed_condition_priors(
            db=db,
            user_id=user_id,
            condition_types=conditions,
        )

    ttl = settings.demo_login_ttl_seconds
    token = _mint_access_token(user_id, email, ttl)

    return DemoLoginResponse(
        access_token=token,
        token_type="bearer",
        user_id=user_id,
        persona=persona,
        expires_in=ttl,
    )
