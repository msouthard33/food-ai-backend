"""reconcile trigger_status_enum to the ORM values (suspect/probable/confirmed/cleared)

Prod's ``trigger_status_enum`` was mis-bootstrapped (create_all + stamp) with the
legacy label set ``{suspected, confirmed, ruled_out}``, but the ORM ``TriggerStatus``
pins ``{suspect, probable, confirmed, cleared}``. Every write of ``status='suspect'``
(seed_condition_priors, update_trigger_predictions) therefore raises
"invalid input value for enum trigger_status_enum" and 500s
``POST /api/v1/users/me/seed-priors``.

``trigger_predictions`` is empty in prod, so the type can be recreated cleanly:
create a correctly-valued type, drop the column default, swap the column type with a
USING cast that maps any legacy rows (suspected->suspect, ruled_out->cleared), drop
the old type, rename the new one into place, and restore the ``'suspect'`` server
default (matching the initial-schema migration's intent).

Guarded/idempotent: it inspects the live label set first and no-ops when the type
already matches the target exactly (e.g. a fresh create_all test DB), so it is safe
on both prod (old labels) and model-derived schemas (already-correct labels). The
whole reconciliation runs inside Alembic's transaction (no bare ``ALTER TYPE ADD
VALUE``, which cannot run transactionally).

Revision ID: n0c1d2e3f4g5
Revises: m9b0c1d2e3f4
Create Date: 2026-07-25
"""

from collections.abc import Sequence

from alembic import op

revision: str = "n0c1d2e3f4g5"
down_revision: str | None = "m9b0c1d2e3f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Reconcile only when the live label set differs from the target set. The USING
# CASE maps legacy labels; because the table is empty in prod the cast touches zero
# rows, but it keeps the migration correct even if rows existed.
_UPGRADE_SQL = """
DO $$
DECLARE
  current_labels text[];
  target_labels  text[] := ARRAY['suspect','probable','confirmed','cleared'];
BEGIN
  SELECT array_agg(e.enumlabel ORDER BY e.enumlabel)
    INTO current_labels
  FROM pg_type t
  JOIN pg_enum e ON e.enumtypid = t.oid
  WHERE t.typname = 'trigger_status_enum';

  -- Only act when the type exists AND its labels don't already match the target.
  IF current_labels IS NOT NULL
     AND current_labels IS DISTINCT FROM (SELECT array_agg(x ORDER BY x)
                                          FROM unnest(target_labels) AS x) THEN

    CREATE TYPE trigger_status_enum_new AS ENUM
      ('suspect', 'probable', 'confirmed', 'cleared');

    ALTER TABLE public."trigger_predictions" ALTER COLUMN status DROP DEFAULT;

    ALTER TABLE public."trigger_predictions"
      ALTER COLUMN status TYPE trigger_status_enum_new
      USING (
        CASE status::text
          WHEN 'suspected' THEN 'suspect'
          WHEN 'ruled_out' THEN 'cleared'
          ELSE status::text
        END
      )::trigger_status_enum_new;

    DROP TYPE trigger_status_enum;
    ALTER TYPE trigger_status_enum_new RENAME TO trigger_status_enum;

    ALTER TABLE public."trigger_predictions"
      ALTER COLUMN status SET DEFAULT 'suspect';
  END IF;
END $$;
"""


def upgrade() -> None:
    op.execute(_UPGRADE_SQL)


def downgrade() -> None:
    # No safe/meaningful downgrade: the legacy label set ({suspected, confirmed,
    # ruled_out}) has no value for 'probable' and cannot represent 'suspect'
    # exactly, so reverting would risk data loss / cast failures. The forward
    # reconciliation is idempotent, so a downgrade is intentionally a no-op.
    pass
