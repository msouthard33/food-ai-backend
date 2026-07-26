"""Wave 2 Pillar 2: capture-side precise timestamps (additive, non-breaking)

Adds capture-precision metadata to meal and symptom log events so the future
exposure/lag rewire has higher-fidelity input. Nothing here restructures the
exposure/event schema or touches correlation queries — it is purely additive.

New columns (all nullable / defaulted so existing rows and existing clients
keep working):

* ``meals.client_timezone``     — IANA zone / UTC offset the client captured the
  meal time in (``timestamp`` is stored UTC-normalised).
* ``meals.time_precision``      — "exact" | "approximate" capture-confidence flag
  (server default 'exact').
* ``symptom_scores.onset_at``   — when the symptom *began*, distinct from
  ``timestamp`` (when observed/logged) and ``created_at`` (server insert).
* ``symptom_scores.client_timezone`` / ``symptom_scores.time_precision`` — as above.

The existing ``timestamp`` (occurred-at) and ``created_at`` (logged-at) columns
are untouched: no drops, no renames, no type changes.

All statements use ``IF NOT EXISTS`` because the prod DB was mis-bootstrapped
(create_all + stamp) and may drift; this keeps the migration idempotent.

Revision ID: p2e3f4a5b6c7
Revises: o1d2e3f4a5b6
Create Date: 2026-07-26
"""

from collections.abc import Sequence

from alembic import op

revision: str = "p2e3f4a5b6c7"
down_revision: str | None = "o1d2e3f4a5b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (table, column_name, column_type_ddl)
_COLUMNS = [
    ('public."meals"', "client_timezone", "VARCHAR(64)"),
    ('public."meals"', "time_precision", "VARCHAR(16) DEFAULT 'exact'"),
    ('public."symptom_scores"', "onset_at", "TIMESTAMPTZ"),
    ('public."symptom_scores"', "client_timezone", "VARCHAR(64)"),
    ('public."symptom_scores"', "time_precision", "VARCHAR(16) DEFAULT 'exact'"),
]


def upgrade() -> None:
    for table, name, ddl in _COLUMNS:
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {ddl}")


def downgrade() -> None:
    for table, name, _ddl in reversed(_COLUMNS):
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {name}")
