"""Ingestion upsert must key on the stable kb_id, so a KB rename updates the
existing row in place instead of orphaning it and inserting a duplicate.

This is the regression test for the 2026-07-26 prod dedup (bare 'Edamame'/'Kimchi'
orphans left behind by the old name-keyed upsert).
"""

import json
import uuid

import pytest
from sqlalchemy import func, select

from app.database import async_session_factory
from app.models.food import FoodEntry
from app.services.food_ingestion import ingest_allergen_knowledge_base

from tests.conftest import _ensure_tables


def _write_kb(tmp_path, foods):
    p = tmp_path / "kb.json"
    p.write_text(json.dumps({"version": "test", "foods": foods}))
    return str(p)


@pytest.mark.asyncio
async def test_rename_matches_on_kb_id_no_orphan(tmp_path):
    await _ensure_tables()
    kb_id = f"test_{uuid.uuid4().hex[:8]}"
    orig_name = f"Zeta Orig {uuid.uuid4().hex[:6]}"
    new_name = f"Zeta Renamed {uuid.uuid4().hex[:6]}"

    path1 = _write_kb(tmp_path, [{"id": kb_id, "name": orig_name, "category": "test"}])
    async with async_session_factory() as db:
        await ingest_allergen_knowledge_base(db, json_path=path1)

    # Re-ingest with the SAME kb_id but a new name (a rename).
    path2 = _write_kb(tmp_path, [{"id": kb_id, "name": new_name, "category": "test"}])
    async with async_session_factory() as db:
        await ingest_allergen_knowledge_base(db, json_path=path2)

    async with async_session_factory() as db:
        rows = (
            await db.execute(select(FoodEntry).where(FoodEntry.kb_id == kb_id))
        ).scalars().all()
    # Exactly one row for this kb_id, and it carries the NEW name (updated in place).
    assert len(rows) == 1, "rename must not create a second row for the same kb_id"
    assert rows[0].name == new_name


@pytest.mark.asyncio
async def test_kb_id_backfilled_on_ingest(tmp_path):
    await _ensure_tables()
    kb_id = f"test_{uuid.uuid4().hex[:8]}"
    name = f"Kappa Food {uuid.uuid4().hex[:6]}"
    path = _write_kb(tmp_path, [{"id": kb_id, "name": name, "category": "test"}])
    async with async_session_factory() as db:
        n = await ingest_allergen_knowledge_base(db, json_path=path)
    assert n == 1
    async with async_session_factory() as db:
        row = (
            await db.execute(select(FoodEntry).where(FoodEntry.name == name))
        ).scalar_one()
    assert row.kb_id == kb_id
