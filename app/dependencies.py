"""FastAPI dependencies — auth, pagination, etc."""

import uuid
from functools import lru_cache

import httpx
from fastapi import Depends, Header, HTTPException, Query, status
from jose import JWTError, jwt
from jose.backends import ECKey
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.user import User

settings = get_settings()


@lru_cache(maxsize=1)
def _get_jwks_keys() -> list[dict]:
    """Fetch and cache JWKS from Supabase (synchronous, called once at startup)."""
    if not settings.supabase_url:
        return []
    jwks_url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
    try:
        resp = httpx.get(jwks_url, timeout=10)
        resp.raise_for_status()
        return resp.json().get("keys", [])
    except Exception:
        return []


def _decode_supabase_jwt(token: str) -> dict:
    """Decode a Supabase JWT, supporting both HS256 and ES256 algorithms.

    New Supabase projects use ES256 (asymmetric) by default. Older projects
    and projects that have not migrated still use HS256 with supabase_jwt_secret.

    Strategy:
      1. If supabase_jwt_secret is set, attempt HS256 decode first.
      2. If that fails (or secret not set), fetch JWKS and try ES256.
    """
    last_error: Exception | None = None

    # Attempt 1: HS256 with the configured JWT secret (legacy / HS256 projects)
    if settings.supabase_jwt_secret:
        try:
            return jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
            )
        except JWTError as exc:
            last_error = exc

    # Attempt 2: ES256 via JWKS (new Supabase projects using asymmetric keys)
    keys = _get_jwks_keys()
    if keys:
        # Try each key (usually only one, but spec allows multiple)
        for key_data in keys:
            try:
                return jwt.decode(
                    token,
                    key_data,
                    algorithms=["ES256"],
                    audience="authenticated",
                )
            except JWTError as exc:
                last_error = exc
                continue

    if last_error:
        raise last_error
    raise JWTError("No valid JWT verification method configured")


async def get_current_user(
    authorization: str = Header(..., description="Bearer <supabase-jwt>"),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Validate Supabase JWT (HS256 or ES256) and return the authenticated user.

    In development mode with no JWT secret configured and no supabase_url,
    accepts a raw UUID for easy testing (e.g., Authorization: Bearer <user-uuid>).
    """
    token = authorization.removeprefix("Bearer ").strip()

    # Dev shortcut: if neither JWT secret nor Supabase URL is configured, accept a raw UUID
    if not settings.supabase_jwt_secret and not settings.supabase_url:
        try:
            user_id = uuid.UUID(token)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user

    # Production: decode Supabase JWT (HS256 or ES256)
    try:
        payload = _decode_supabase_jwt(token)
        sub = payload.get("sub")
        if sub is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        user_id = uuid.UUID(sub)
    except (JWTError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        # Auto-provision user from Supabase Auth on first API call
        user = User(id=user_id, email=payload.get("email", ""))
        db.add(user)
        await db.flush()

    return user


class PaginationParams:
    """Reusable pagination query parameters."""

    def __init__(
        self,
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    ):
        self.page = page
        self.page_size = page_size
        self.offset = (page - 1) * page_size
