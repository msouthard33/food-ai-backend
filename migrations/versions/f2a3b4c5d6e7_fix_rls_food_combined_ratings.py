"""fix RLS gap: enable RLS on food_combined_ratings (BLK-SEC-001 follow-up)

Migration b3d1e2f4a5c8 created food_combined_ratings without RLS.
Migration e8a4c7b2f1d9 added RLS to all other Pillar 3 tables but
omitted food_combined_ratings from its ALL_TABLES list.

This migration closes that gap. food_combined_ratings is user-owned
(rows belong to a specific user_id) so it gets default-deny RLS with
no explicit policies — the FastAPI backend connects via the postgres
role which has BYPASSRLS, so backend behavior is unchanged.

Hard rule satisfied: every public table now has ENABLE + FORCE RLS.

Revision ID: f2a3b4c5d6e7
Revises: e8a4c7b2f1d9
Create Date: 2026-04-11
"""
from typing import Sequence, Union

from alembic import op

revision: str = "f2a3b4c5d6e7"
down_revision: Union[str, None] = "e8a4c7b2f1d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "food_combined_ratings"


def upgrade() -> None:
    op.execute(f'ALTER TABLE IF EXISTS public."{TABLE}" ENABLE ROW LEVEL SECURITY;')
    op.execute(f'ALTER TABLE IF EXISTS public."{TABLE}" FORCE ROW LEVEL SECURITY;')


def downgrade() -> None:
    op.execute(f'ALTER TABLE IF EXISTS public."{TABLE}" NO FORCE ROW LEVEL SECURITY;')
    op.execute(f'ALTER TABLE IF EXISTS public."{TABLE}" DISABLE ROW LEVEL SECURITY;')
