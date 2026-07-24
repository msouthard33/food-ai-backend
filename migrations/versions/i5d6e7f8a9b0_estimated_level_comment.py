"""Add PostgreSQL column comment to meal_item_components.estimated_level.

Documents the 0-100 allergen load score convention and cumulative daily
load aggregation semantics so future engineers and analytics queries have
an authoritative reference at the column level.

Revision ID: i5d6e7f8a9b0
Revises: h4c5d6e7f8a9
Create Date: 2026-07-23 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "i5d6e7f8a9b0"
down_revision: Union[str, None] = "h4c5d6e7f8a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "COMMENT ON COLUMN meal_item_components.estimated_level IS "
        "'Allergen load score 0-100. Sum within ComponentType across a day for "
        "cumulative load analysis. Source: KB allergen_profile scores.'"
    )


def downgrade() -> None:
    op.execute("COMMENT ON COLUMN meal_item_components.estimated_level IS NULL")
