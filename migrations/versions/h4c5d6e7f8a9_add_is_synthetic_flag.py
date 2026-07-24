"""Add is_synthetic flag to users table.

Supports cold-start synthetic cohort — allows the system to distinguish
synthetic population-prior patients from real users and exclude them from
personalized trigger logic once the decay threshold is reached.

Revision ID: h4c5d6e7f8a9
Revises: g3b4c5d6e7f8
Create Date: 2026-07-23 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "h4c5d6e7f8a9"
down_revision: Union[str, None] = "g3b4c5d6e7f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "is_synthetic",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.create_index("ix_users_is_synthetic", "users", ["is_synthetic"])


def downgrade() -> None:
    op.drop_index("ix_users_is_synthetic", table_name="users")
    op.drop_column("users", "is_synthetic")
