"""enable RLS on all public tables (BLK-SEC-001)

Resolves Supabase advisor finding rls_disabled_in_public (2026-04-07).

Strategy:
- Enable + FORCE row level security on every public.* app table.
- Reference tables (food_database, food_component_details, component_definitions,
  barcode_product_cache) get an anon/authenticated SELECT policy (public read).
- All user-owned and internal tables get RLS enabled with NO policies, meaning
  default-deny for anon/authenticated. The FastAPI backend writes via the
  Supabase service role key, which bypasses RLS, so backend behavior is
  unchanged.
- Wave 2 / Pillar 3 hardening.

Revision ID: e8a4c7b2f1d9
Revises: d7f9a2b1c4e6
Create Date: 2026-04-08
"""
from typing import Sequence, Union

from alembic import op


revision: str = "e8a4c7b2f1d9"
down_revision: Union[str, None] = "d7f9a2b1c4e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# All app-owned tables in the public schema.
#
# Railway deployment note (2026-04-09, Track 2): the FastAPI backend connects
# with DATABASE_URL using the `postgres` role. That role has been granted
# BYPASSRLS out-of-band so the backend continues to read/write user-owned
# tables normally — RLS here is defense-in-depth against a leaked
# non-superuser role, mirroring the Supabase service-role bypass model.
ALL_TABLES = [
    # user-owned (default-deny, app role bypasses via BYPASSRLS grant)
    "users",
    "user_settings",
    "user_conditions",
    "user_known_allergens",
    "user_sensitivity_profiles",
    "meals",
    "meal_items",
    "meal_item_components",
    "ai_conversations",
    "insight_reports",
    "report_shares",
    "symptom_scores",
    "daily_checkins",
    "trigger_predictions",
    "correlation_events",
    # public reference (public SELECT policy added below)
    "food_database",
    "food_component_details",
    "component_definitions",
    "barcode_product_cache",
    "food_embeddings",
    "food_embeddings_oai",
]

PUBLIC_READ_TABLES = [
    "food_database",
    "food_component_details",
    "component_definitions",
    "barcode_product_cache",
    "food_embeddings",
    "food_embeddings_oai",
]


def upgrade() -> None:
    for t in ALL_TABLES:
        op.execute(f'ALTER TABLE IF EXISTS public."{t}" ENABLE ROW LEVEL SECURITY;')
        op.execute(f'ALTER TABLE IF EXISTS public."{t}" FORCE ROW LEVEL SECURITY;')

    for t in PUBLIC_READ_TABLES:
        policy = f"{t}_public_read"
        op.execute(
            f'DROP POLICY IF EXISTS "{policy}" ON public."{t}";'
        )
        op.execute(
            f'CREATE POLICY "{policy}" ON public."{t}" '
            f"FOR SELECT USING (true);"
        )


def downgrade() -> None:
    for t in PUBLIC_READ_TABLES:
        policy = f"{t}_public_read"
        op.execute(f'DROP POLICY IF EXISTS "{policy}" ON public."{t}";')

    for t in ALL_TABLES:
        op.execute(f'ALTER TABLE IF EXISTS public."{t}" NO FORCE ROW LEVEL SECURITY;')
        op.execute(f'ALTER TABLE IF EXISTS public."{t}" DISABLE ROW LEVEL SECURITY;')
