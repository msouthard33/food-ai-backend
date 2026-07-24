"""W2-B1: soft-delete columns + perf indexes for export & insights leaderboard

Adds a nullable ``deleted_at`` column to the user-owned tables (meals, symptom_scores,
medication_logs, user_sensitivity_profiles) so rows can be soft-deleted and excluded
from insights and the full data export, and adds composite ``(user_id, <time>)``
indexes that accelerate the per-user, time-ranged scans those features run.

All statements use ``IF NOT EXISTS`` / ``IF EXISTS`` because the prod DB was
mis-bootstrapped (create_all + stamp) and may drift from the model-derived schema;
this keeps the migration idempotent and safe to re-run.

Revision ID: m9b0c1d2e3f4
Revises: l8a9b0c1d2e3
Create Date: 2026-07-24
"""

from typing import Sequence, Union

from alembic import op

revision: str = "m9b0c1d2e3f4"
down_revision: Union[str, None] = "l8a9b0c1d2e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# table -> soft-delete column
_SOFT_DELETE_TABLES = [
    "meals",
    "symptom_scores",
    "medication_logs",
    "user_sensitivity_profiles",
]

# (index_name, table, columns) — mirror the model __table_args__ definitions
_INDEXES = [
    ("ix_meals_user_timestamp", "meals", "user_id, timestamp"),
    ("ix_symptom_scores_user_timestamp", "symptom_scores", "user_id, timestamp"),
    ("ix_medication_logs_user_taken_at", "medication_logs", "user_id, taken_at"),
]


def upgrade() -> None:
    for table in _SOFT_DELETE_TABLES:
        op.execute(
            f'ALTER TABLE public."{table}" '
            f"ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE"
        )

    for index_name, table, columns in _INDEXES:
        op.execute(
            f'CREATE INDEX IF NOT EXISTS {index_name} ON public."{table}" ({columns})'
        )


def downgrade() -> None:
    for index_name, table, _columns in _INDEXES:
        op.execute(f"DROP INDEX IF EXISTS {index_name}")

    for table in _SOFT_DELETE_TABLES:
        op.execute(f'ALTER TABLE public."{table}" DROP COLUMN IF EXISTS deleted_at')
