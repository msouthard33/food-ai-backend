"""use component_type_enum (not legacy 'componenttype') on component_type columns

food_component_details.component_type and component_definitions.component_type were
bootstrapped with SQLAlchemy's auto-named `componenttype` enum type, but the models
pin `name="component_type_enum"`. Queries therefore cast to component_type_enum and
Postgres raises: "operator does not exist: componenttype = component_type_enum".
Both tables are empty in prod, so the USING cast converts zero rows. Idempotent:
only runs when the column is still the legacy `componenttype` type.

Revision ID: l8a9b0c1d2e3
Revises: k7f8a9b0c1d2
Create Date: 2026-07-24
"""

from typing import Sequence, Union

from alembic import op

revision: str = "l8a9b0c1d2e3"
down_revision: Union[str, None] = "k7f8a9b0c1d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = ["food_component_details", "component_definitions"]


def _fix(table: str) -> str:
    return f"""
    DO $$
    BEGIN
      IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='{table}'
              AND column_name='component_type' AND udt_name='componenttype'
      ) THEN
        ALTER TABLE public."{table}"
          ALTER COLUMN component_type TYPE component_type_enum
          USING component_type::text::component_type_enum;
      END IF;
    END $$;
    """


def upgrade() -> None:
    for t in _TABLES:
        op.execute(_fix(t))


def downgrade() -> None:
    # No safe/meaningful downgrade: the legacy `componenttype` type may lack values
    # present in component_type_enum. Intentionally a no-op.
    pass
