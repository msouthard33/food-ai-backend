"""Wave 2 Sprint H4: persist hierarchical-Bayes posteriors + guardrail on trigger_predictions

Wires the hierarchical Bayesian logistic engine (H1) into the persisted trigger
predictions and stores the frequentist FDR guardrail (H2) verdict alongside it (the
"hybrid": Bayesian signal + classical agreement check). ``confidence_score`` keeps its
name/type (now the hierarchical score, trigger_probability*100 on 0–100) so mobile and
the clinician PDF are unaffected; the raw Laplace posterior params (β, SE) + derived
odds-ratio credible interval and the de-confounded ``trigger_probability`` are added so
the score is auditable/reproducible, a ``method`` tag versions the scoring contract, and
``assoc_p_value`` / ``assoc_agreement`` capture the classical guardrail's per-component
p-value and whether it agrees with the Bayesian flag.

All statements use ``IF NOT EXISTS`` because the prod DB was mis-bootstrapped
(create_all + stamp) and may drift from the model-derived schema; this keeps the
migration idempotent and safe to re-run.

Revision ID: o1d2e3f4a5b6
Revises: m9b0c1d2e3f4
Create Date: 2026-07-26
"""

from typing import Sequence, Union

from alembic import op

revision: str = "o1d2e3f4a5b6"
down_revision: Union[str, None] = "m9b0c1d2e3f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = 'public."trigger_predictions"'

# (column_name, column_type_ddl)
_COLUMNS = [
    ("method", "VARCHAR(40) NOT NULL DEFAULT 'hierarchical_bayes_logistic'"),
    ("trigger_probability", "NUMERIC(6, 5)"),
    ("bayes_beta", "NUMERIC(12, 6)"),
    ("bayes_beta_se", "NUMERIC(12, 6)"),
    ("bayes_ci_low", "NUMERIC(18, 6)"),
    ("bayes_ci_high", "NUMERIC(18, 6)"),
    ("assoc_p_value", "NUMERIC(10, 8)"),
    ("assoc_agreement", "BOOLEAN"),
]


def upgrade() -> None:
    for name, ddl in _COLUMNS:
        op.execute(f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS {name} {ddl}")


def downgrade() -> None:
    for name, _ddl in reversed(_COLUMNS):
        op.execute(f"ALTER TABLE {_TABLE} DROP COLUMN IF EXISTS {name}")
