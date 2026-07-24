"""align food_component_details columns to the ORM (food_id, level_score)

Prod's food_component_details was bootstrapped from an OLDER model revision whose
columns were `food_entry_id` and `level`. The models were later refactored to
`food_id` (DB name for the food_entry_id attribute) and `level_score` (DB name for
the level attribute), but prod's schema was never actually migrated — alembic_version
was stamped to head, so `alembic upgrade head` is a no-op. Result: every query
against the table 500s with "column food_component_details.food_id does not exist",
and the table (and food_database) can never be populated.

This renames the two columns to match the ORM. Idempotent: only renames when the
old column exists and the new one does not, so it is safe on already-correct DBs
(e.g. a fresh create_all environment).

Verified against prod: food_component_details is empty (0 rows) and the extra
`source` column is NOT NULL but has a default, so ORM inserts succeed after this.

Revision ID: k7f8a9b0c1d2
Revises: j6e7f8a9b0c1
Create Date: 2026-07-24
"""

from typing import Sequence, Union

from alembic import op

revision: str = "k7f8a9b0c1d2"
down_revision: Union[str, None] = "j6e7f8a9b0c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "food_component_details"


def _rename_if_needed(old: str, new: str) -> str:
    return f"""
    DO $$
    BEGIN
      IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='{_TABLE}' AND column_name='{old}'
      ) AND NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='{_TABLE}' AND column_name='{new}'
      ) THEN
        ALTER TABLE public."{_TABLE}" RENAME COLUMN "{old}" TO "{new}";
      END IF;
    END $$;
    """


def upgrade() -> None:
    op.execute(_rename_if_needed("food_entry_id", "food_id"))
    op.execute(_rename_if_needed("level", "level_score"))


def downgrade() -> None:
    op.execute(_rename_if_needed("food_id", "food_entry_id"))
    op.execute(_rename_if_needed("level_score", "level"))
