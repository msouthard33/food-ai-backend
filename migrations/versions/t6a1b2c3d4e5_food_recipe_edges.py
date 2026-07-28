"""ADR-0003 §4 — food_recipe_edges (recipe/composition graph) + RLS + curated seed

Adds the structured edge layer from a composite dish to its constituent KB
foods. This is the deterministic "curated" branch of the unified breakdown
contract (ADR-0003 §1/§2). Additive — it never replaces a food's
``allergen_profile``.

Guardrails (CLAUDE.md): this is a new ``public.*`` table, so ENABLE + FORCE
ROW LEVEL SECURITY and explicit policies are created **in this same migration**.
``food_recipe_edges`` is public, non-sensitive reference data (a recipe graph,
no PHI), reached only through the app's single DB role — so it gets a public
SELECT policy plus an app-write (FOR ALL) policy, mirroring the other reference
tables (food_database, barcode_product_cache, …) per migrations e8a4c7b2f1d9 /
j6e7f8a9b0c1.

``provenance`` is a CHECK-constrained String (values 'curated','promoted'),
not a PG ENUM type, so the table builds identically under Alembic and under
Base.metadata.create_all (the test DB does not provision standalone enum types).

The curated seed is **FK-safe and idempotent**: each edge is inserted via
INSERT…SELECT that resolves both the composite and the ingredient by name
against ``food_database`` and only writes when *both* exist (ON CONFLICT DO
NOTHING). On an empty/dev DB it seeds nothing; on the populated prod KB it
seeds a tiny first curated set for a few top composite dishes. Deferred build
step 5 (the promotion loop) widens this over time.

APPLIED TO WORKING/DEV DB ONLY. Prod migration + deploy is a HUMAN GATECHECK
owned by the orchestrator/Matt (see the sprint report next_actions).

Revision ID: t6a1b2c3d4e5
Revises: q3f4a5b6c7d8
Create Date: 2026-07-28
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "t6a1b2c3d4e5"
down_revision: Union[str, None] = "q3f4a5b6c7d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Tiny first curated set (ADR-0003 §4 — seeded from prose `notes` on top
# composites). Keyed by an ILIKE pattern for the composite and each ingredient.
# provenance is always 'curated'. Deliberately small; FK-safe (only inserts
# when both foods exist in the KB).
#   composite_pattern -> [(ingredient_pattern, default_selected, typical_portion, confidence)]
CURATED_SEED: dict[str, list[tuple[str, bool, str, float]]] = {
    "%sushi%": [
        ("%white rice%", True, "1 cup", 0.95),
        ("%nori%", True, "2 sheets", 0.90),
        ("%salmon%", True, "2 oz", 0.85),
        ("%avocado%", False, "1/4 fruit", 0.80),
        ("%cucumber%", False, "1/4 cup", 0.80),
        ("%soy sauce%", False, "1 tbsp", 0.85),
    ],
    "%burrito%": [
        ("%rice%", True, "1/2 cup", 0.95),
        ("%black bean%", True, "1/2 cup", 0.90),
        ("%chicken%", True, "3 oz", 0.85),
        ("%cheese%", True, "1/4 cup", 0.85),
        ("%sour cream%", False, "1 tbsp", 0.80),
        ("%salsa%", False, "2 tbsp", 0.80),
    ],
    "%pizza%": [
        ("%mozzarella%", True, "1/2 cup", 0.90),
        ("%tomato%", True, "1/4 cup", 0.85),
        ("%wheat%", True, "1 slice", 0.80),
    ],
}


def upgrade() -> None:
    op.create_table(
        "food_recipe_edges",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "composite_food_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("food_database.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "ingredient_food_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("food_database.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("default_selected", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("typical_portion", sa.Text),
        sa.Column("provenance", sa.String(16), nullable=False, server_default="curated"),
        sa.Column("confidence", sa.Numeric(4, 3)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "composite_food_id",
            "ingredient_food_id",
            name="uq_food_recipe_edges_composite_ingredient",
        ),
        sa.CheckConstraint(
            "provenance IN ('curated', 'promoted')",
            name="ck_food_recipe_edges_provenance",
        ),
    )
    op.create_index(
        "ix_food_recipe_edges_composite_food_id",
        "food_recipe_edges",
        ["composite_food_id"],
    )
    op.create_index(
        "ix_food_recipe_edges_ingredient_food_id",
        "food_recipe_edges",
        ["ingredient_food_id"],
    )

    # --- RLS in the SAME migration (CLAUDE.md guardrail) ---
    op.execute('ALTER TABLE public."food_recipe_edges" ENABLE ROW LEVEL SECURITY;')
    op.execute('ALTER TABLE public."food_recipe_edges" FORCE ROW LEVEL SECURITY;')
    op.execute('DROP POLICY IF EXISTS "food_recipe_edges_public_read" ON public."food_recipe_edges";')
    op.execute(
        'CREATE POLICY "food_recipe_edges_public_read" ON public."food_recipe_edges" '
        "FOR SELECT USING (true);"
    )
    op.execute('DROP POLICY IF EXISTS "food_recipe_edges_app_write" ON public."food_recipe_edges";')
    op.execute(
        'CREATE POLICY "food_recipe_edges_app_write" ON public."food_recipe_edges" '
        "FOR ALL USING (true) WITH CHECK (true);"
    )

    # --- Tiny curated seed (FK-safe, idempotent) ---
    for composite_pat, edges in CURATED_SEED.items():
        for ingredient_pat, default_selected, portion, confidence in edges:
            op.execute(
                sa.text(
                    """
                    INSERT INTO food_recipe_edges
                        (id, composite_food_id, ingredient_food_id,
                         default_selected, typical_portion, provenance, confidence)
                    SELECT gen_random_uuid(), c.id, i.id, :sel, :portion, 'curated', :conf
                    FROM (SELECT id FROM food_database WHERE name ILIKE :cpat
                          ORDER BY name LIMIT 1) c
                    CROSS JOIN (SELECT id FROM food_database WHERE name ILIKE :ipat
                                ORDER BY name LIMIT 1) i
                    WHERE c.id <> i.id
                    ON CONFLICT (composite_food_id, ingredient_food_id) DO NOTHING;
                    """
                ).bindparams(
                    sel=default_selected,
                    portion=portion,
                    conf=confidence,
                    cpat=composite_pat,
                    ipat=ingredient_pat,
                )
            )


def downgrade() -> None:
    op.execute('DROP POLICY IF EXISTS "food_recipe_edges_app_write" ON public."food_recipe_edges";')
    op.execute('DROP POLICY IF EXISTS "food_recipe_edges_public_read" ON public."food_recipe_edges";')
    op.drop_index(
        "ix_food_recipe_edges_ingredient_food_id", table_name="food_recipe_edges"
    )
    op.drop_index(
        "ix_food_recipe_edges_composite_food_id", table_name="food_recipe_edges"
    )
    op.drop_table("food_recipe_edges")
