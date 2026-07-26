"""Add stable kb_id business key to food_database + backfill from KB JSON.

Root-cause fix for the 2026-07-26 prod dedup: the ingest upsert matched foods on
``name`` because the table had no stable KB identifier (the PK is a random uuid).
When the 2026-07-24 KB dedup renamed entries (e.g. 'Edamame' -> 'Edamame (raw
soybeans)'), the name-keyed upsert inserted a *new* row and orphaned the old one,
pushing prod to 1502 rows instead of the KB's 1500.

This migration adds a nullable ``kb_id VARCHAR`` column (b-tree index + a partial
UNIQUE index for non-null values) and backfills it by matching each existing
``food_database.name`` to the KB entry ``name`` -> KB ``id`` in
``data/allergen_knowledge_base_complete.json`` (v2.7.0, 1500 entries).

All DDL uses IF NOT EXISTS because the prod DB was mis-bootstrapped (create_all +
stamp) and can drift from the model-derived schema; this keeps the migration
idempotent and safe to re-run.

Revision ID: q3f4a5b6c7d8
Revises: p2e3f4a5b6c7
Create Date: 2026-07-26
"""

import json
from pathlib import Path
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "q3f4a5b6c7d8"
down_revision: Union[str, None] = "p2e3f4a5b6c7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# backend/data/allergen_knowledge_base_complete.json — versions/<file>.parents:
# [0]=versions, [1]=migrations, [2]=backend
_KB_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "allergen_knowledge_base_complete.json"
)


def _load_name_to_kb_id() -> dict[str, str]:
    if not _KB_PATH.exists():
        return {}
    with _KB_PATH.open() as fh:
        data = json.load(fh)
    records = data.get("foods", []) if isinstance(data, dict) else data
    mapping: dict[str, str] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        name = (rec.get("name") or "").strip()
        kb_id = (rec.get("id") or "").strip()
        if name and kb_id:
            mapping[name] = kb_id
    return mapping


def upgrade() -> None:
    op.execute(
        'ALTER TABLE public."food_database" '
        "ADD COLUMN IF NOT EXISTS kb_id VARCHAR(64)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_food_database_kb_id "
        'ON public."food_database" (kb_id)'
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_food_database_kb_id "
        'ON public."food_database" (kb_id) WHERE kb_id IS NOT NULL'
    )

    # Backfill kb_id by matching name -> KB id. Rows whose name has no KB match
    # stay NULL (legacy / non-KB rows). executemany over a parameterized UPDATE.
    mapping = _load_name_to_kb_id()
    if mapping:
        bind = op.get_bind()
        params = [{"kb_id": kb_id, "name": name} for name, kb_id in mapping.items()]
        bind.execute(
            text(
                'UPDATE public."food_database" SET kb_id = :kb_id '
                "WHERE name = :name AND kb_id IS NULL"
            ),
            params,
        )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_food_database_kb_id")
    op.execute("DROP INDEX IF EXISTS ix_food_database_kb_id")
    op.execute('ALTER TABLE public."food_database" DROP COLUMN IF EXISTS kb_id')
