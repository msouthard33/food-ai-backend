"""allow app writes to public reference tables (ingest unblock)

Migration e8a4c7b2f1d9 put FORCE row-level security + a SELECT-only policy on the
public reference tables. Reads work, but INSERT/UPDATE are blocked unless the app
role has BYPASSRLS — a grant the RLS migration expected to be applied "out-of-band"
but which is not in effect in prod. Result: POST /admin/ingest-foods 500s and
food_database stays empty.

This adds a permissive write policy (FOR ALL) to the *reference* tables only, so
ingestion + barcode/embedding caches work without BYPASSRLS. These tables hold
public, non-sensitive reference data and are only reachable through the app's
single DB role on Railway (there is no anon/authenticated DB path), so USING(true)
is acceptable here.

DELIBERATELY NOT TOUCHED: user-owned tables (users, meals, symptom_scores, ...)
remain default-deny. They must NOT get USING(true) write policies (that would let
any role read/write all users' data). Those still require the app role's BYPASSRLS
grant before launch — a separate prod DB-admin action. Once BYPASSRLS is granted,
these reference write policies become redundant and may be dropped.

Revision ID: j6e7f8a9b0c1
Revises: i5d6e7f8a9b0
Create Date: 2026-07-24
"""

from typing import Sequence, Union

from alembic import op

revision: str = "j6e7f8a9b0c1"
down_revision: Union[str, None] = "i5d6e7f8a9b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Public reference tables (mirrors PUBLIC_READ_TABLES in e8a4c7b2f1d9).
REFERENCE_TABLES = [
    "food_database",
    "food_component_details",
    "component_definitions",
    "barcode_product_cache",
    "food_embeddings",
    "food_embeddings_oai",
]


def upgrade() -> None:
    for t in REFERENCE_TABLES:
        policy = f"{t}_app_write"
        op.execute(f'DROP POLICY IF EXISTS "{policy}" ON public."{t}";')
        op.execute(
            f'CREATE POLICY "{policy}" ON public."{t}" '
            f"FOR ALL USING (true) WITH CHECK (true);"
        )


def downgrade() -> None:
    for t in REFERENCE_TABLES:
        op.execute(f'DROP POLICY IF EXISTS "{t}_app_write" ON public."{t}";')
