"""Bayesian Sprint 3: persist Beta-Binomial posteriors on trigger_predictions

Wires the Beta-Binomial engine into the persisted trigger predictions. ``confidence_score``
keeps its name/type (now the Bayesian score, trigger_probability*100 on 0–100) so mobile
and the clinician PDF are unaffected; the raw posterior params + derived credible interval
and the de-confounded ``trigger_probability`` are added alongside so the score is auditable
and reproducible, and a ``method`` tag versions the scoring contract.

All statements use ``IF NOT EXISTS`` because the prod DB was mis-bootstrapped
(create_all + stamp) and may drift from the model-derived schema; this keeps the
migration idempotent and safe to re-run.

Revision ID: n0c1d2e3f4a5
Revises: m9b0c1d2e3f4
Create Date: 2026-07-25
"""

from typing import Sequence, Union

from alembic import op

revision: str = "n0c1d2e3f4a5"
down_revision: Union[str, None] = "m9b0c1d2e3f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = 'public."trigger_predictions"'

# (column_name, column_type_ddl)
_COLUMNS = [
    ("method", "VARCHAR(40) NOT NULL DEFAULT 'bayesian_beta_binomial'"),
    ("trigger_probability", "NUMERIC(6, 5)"),
    ("bayesian_alpha", "NUMERIC(10, 4)"),
    ("bayesian_beta", "NUMERIC(10, 4)"),
    ("bayesian_ci_low", "NUMERIC(5, 2)"),
    ("bayesian_ci_high", "NUMERIC(5, 2)"),
]


def upgrade() -> None:
    for name, ddl in _COLUMNS:
        op.execute(f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS {name} {ddl}")


def downgrade() -> None:
    for name, _ddl in reversed(_COLUMNS):
        op.execute(f"ALTER TABLE {_TABLE} DROP COLUMN IF EXISTS {name}")
