"""FastAPI dependencies — auth, pagination, etc."""

import uuid

from fastapi import Depends, Header, HTTPException, Query, status
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.user import User

settings = get_settings()


async def get_current_user(
    authorization: str = Header(..., description="Bearer <supabase-jwt>"),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Validate Supabase JWT and return the authenticated user.

    In development mode with no JWT secret configured, accepts a raw UUID
    for easy testing (e.g., Authorization: Bearer <user-uuid>).
    """
    token = authorization.removeprefix("Bearer ").strip()

    # Dev shortcut: if no JWT secret is set, treat token as user UUID
    if not settings.supabase_jwt_secret:
        try:
            user_id = uuid.UUID(token)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user

    # Production: decode Supabase JWT
    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
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
