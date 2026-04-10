"""add_missing_component_types

Adds nickel, bromelain, and nitrates to component_type_enum.
These values are present in the Python ComponentType enum (enums.py)
but were omitted from the initial schema migration. Required to store
nickel sensitivity data for 408 foods in the KB.

Revision ID: a2c9f0e1d3b7
Revises: f051b1bff6d4
Create Date: 2026-04-06 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a2c9f0e1d3b7"
down_revision: Union[str, None] = "f051b1bff6d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add the three missing values to component_type_enum.
    # PostgreSQL requires ALTER TYPE ... ADD VALUE to run outside a transaction
    # when using psycopg2 in autocommit mode; Alembic handles this via
    # execute_if / migration_context.
    op.execute("ALTER TYPE component_type_enum ADD VALUE IF NOT EXISTS 'nickel'")
    op.execute("ALTER TYPE component_type_enum ADD VALUE IF NOT EXISTS 'bromelain'")
    op.execute("ALTER TYPE component_type_enum ADD VALUE IF NOT EXISTS 'nitrates'")


def downgrade() -> None:
    # PostgreSQL does not support removing enum values without recreating the type.
    # To downgrade, manually recreate the enum without these values and update
    # any rows referencing them first.
    raise NotImplementedError(
        "Downgrade not supported for enum value removal. "
        "Manually recreate component_type_enum without nickel/bromelain/nitrates."
    )
