#!/usr/bin/env python3
"""Generate OpenAI embeddings for all KB foods and insert into food_embeddings_oai.

This script:
1. Loads the KB JSON
2. Queries the Railway PG food_database for food_id by name
3. Generates OpenAI text-embedding-3-small embeddings for each food
4. Inserts (or upserts) into food_embeddings_oai

Prerequisites:
  - OPENAI_API_KEY set in environment
  - DATABASE_PUBLIC_URL or RAILWAY_PG_URL set (Railway PG external URL)
  - food_database table populated with KB foods
  - food_embeddings_oai table exists (migration d7f9a2b1c4e6)

Usage:
    OPENAI_API_KEY=sk-... RAILWAY_PG_URL=postgresql://... python backend/scripts/generate_oai_embeddings.py

    # Dry run (compute embeddings, save to JSON, don't write to DB):
    OPENAI_API_KEY=sk-... python backend/scripts/generate_oai_embeddings.py --dry-run
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import openai

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 200

KB_PATH = PROJECT_ROOT / "04 - Food Science & Data" / "allergen_knowledge_base_complete.json"
ARTIFACTS_DIR = PROJECT_ROOT / "04 - Food Science & Data" / "artifacts"


async def embed_batch(client: openai.AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i : i + BATCH_SIZE]
        resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
        all_embs.extend([item.embedding for item in resp.data])
        if i + BATCH_SIZE < len(texts):
            await asyncio.sleep(0.3)
    return all_embs


def build_source_text(food: dict) -> str:
    """Build embedding source text — just the food name (best single-embedding approach)."""
    return food["name"]


async def main(dry_run: bool = False):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        try:
            api_key = Path("/tmp/oai_key.txt").read_text().strip()
        except FileNotFoundError:
            pass
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    client = openai.AsyncOpenAI(api_key=api_key)

    # Load KB
    with open(KB_PATH) as f:
        kb = json.load(f)
    foods = kb["foods"]
    print(f"KB: {len(foods)} foods (v{kb.get('version', '?')})")

    # Generate embeddings
    source_texts = [build_source_text(f) for f in foods]
    print(f"Generating {len(source_texts)} embeddings...")
    t0 = time.time()
    embeddings = await embed_batch(client, source_texts)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(foods)/elapsed:.0f} foods/s)")

    # Also generate alias embeddings (for multi-representation search)
    all_alias_data = []
    for food in foods:
        for alias in food.get("common_names", []) or []:
            if alias and alias.lower() != food["name"].lower():
                all_alias_data.append({"food_name": food["name"], "alias": alias})

    if all_alias_data:
        alias_texts = [a["alias"] for a in all_alias_data]
        print(f"Generating {len(alias_texts)} alias embeddings...")
        alias_embeddings = await embed_batch(client, alias_texts)
        print(f"  Done")
    else:
        alias_embeddings = []

    # Save embeddings to JSON artifact (always, even in dry-run)
    output_data = {
        "artifact_id": "w2-1c_food_embeddings_oai",
        "sprint": "W2-1c",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": EMBEDDING_MODEL,
        "model_version": "2024-02",
        "dimension": EMBEDDING_DIM,
        "kb_version": kb.get("version"),
        "food_count": len(foods),
        "alias_count": len(all_alias_data),
        "foods": [
            {
                "name": food["name"],
                "source_text": source_texts[i],
                "embedding": embeddings[i],
            }
            for i, food in enumerate(foods)
        ],
        "aliases": [
            {
                "food_name": all_alias_data[i]["food_name"],
                "alias": all_alias_data[i]["alias"],
                "embedding": alias_embeddings[i],
            }
            for i in range(len(all_alias_data))
        ] if alias_embeddings else [],
    }

    out_path = ARTIFACTS_DIR / "w2-1c_food_embeddings_oai.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f)
    print(f"Saved embeddings artifact to {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    if dry_run:
        print("\nDRY RUN — skipping database insert")
        return

    # Insert into Railway PG
    db_url = os.environ.get("RAILWAY_PG_URL") or os.environ.get("DATABASE_PUBLIC_URL", "")
    if not db_url:
        try:
            db_url = Path("/tmp/railway_db_url.txt").read_text().strip()
        except FileNotFoundError:
            pass
    if not db_url:
        print("\nWARNING: No database URL available. Embeddings saved to artifact only.")
        print("To insert into Railway PG, set RAILWAY_PG_URL and re-run without --dry-run.")
        return

    try:
        import psycopg2
    except ImportError:
        print("\nWARNING: psycopg2 not installed. Run: pip install psycopg2-binary")
        print("Embeddings saved to artifact — insert manually or re-run after install.")
        return

    print(f"\nConnecting to Railway PG...")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Check if food_database has data
    cur.execute("SELECT count(*) FROM food_database")
    food_count = cur.fetchone()[0]
    print(f"  food_database rows: {food_count}")

    if food_count == 0:
        print("  WARNING: food_database is empty — cannot map food names to IDs")
        print("  Run food ingestion first, then re-run this script.")
        conn.close()
        return

    # Map food names to IDs
    cur.execute("SELECT id, name FROM food_database")
    food_id_map = {row[1]: str(row[0]) for row in cur.fetchall()}
    print(f"  Mapped {len(food_id_map)} foods to IDs")

    # Insert embeddings
    inserted = 0
    skipped = 0
    for i, food in enumerate(foods):
        food_id = food_id_map.get(food["name"])
        if not food_id:
            skipped += 1
            continue

        emb_str = "[" + ",".join(str(v) for v in embeddings[i]) + "]"
        cur.execute(
            """INSERT INTO food_embeddings_oai (food_id, embedding, model, model_version, source_text)
               VALUES (%s, %s::vector, %s, %s, %s)
               ON CONFLICT (food_id) DO UPDATE SET
                 embedding = EXCLUDED.embedding,
                 model = EXCLUDED.model,
                 model_version = EXCLUDED.model_version,
                 source_text = EXCLUDED.source_text""",
            (food_id, emb_str, EMBEDDING_MODEL, "2024-02", source_texts[i]),
        )
        inserted += 1

    conn.commit()
    print(f"  Inserted: {inserted}, Skipped (no ID match): {skipped}")

    # Verify
    cur.execute("SELECT count(*) FROM food_embeddings_oai")
    embed_count = cur.fetchone()[0]
    print(f"  food_embeddings_oai rows: {embed_count}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Generate embeddings but don't write to DB")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
