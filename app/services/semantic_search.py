"""pgvector-backed semantic food search — W2-1 Pillar 1.

Stores one embedding per food in the `food_embeddings` sidecar table
(migration c4e2f1a9d6b0) and provides similarity search via cosine distance.

Design notes:
- Embeddings are built from a deterministic `source_text` = name + common_names
  + category + subcategory. This keeps embeddings reproducible and lets us
  rebuild idempotently.
- We use `llm_provider.embed_text()` which today returns an offline trigram-
  hash vector. Swapping to a live embedder (OpenAI/Cohere) is a one-line
  change — callers are unaffected.
- Every search result returns a `score` in [0,1] where 1.0 = identical. This
  is `1 - cosine_distance`.
- Cold-start safety: if the embeddings table is empty, the `search` function
  silently falls back to the lexical LIKE search in food_service.search_foods,
  so the API never dead-ends on a fresh DB.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.food import FoodEntry
from app.services import food_service
from app.services.llm_provider import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_ID,
    OPENAI_EMBEDDING_DIM,
    OPENAI_EMBEDDING_MODEL_ID,
    embed_text,
)

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"

# W2-1b: provider-aware sidecar selection.
#   "offline" -> food_embeddings (384-dim, trigram-hash, W2-1)
#   "openai"  -> food_embeddings_oai (1536-dim, text-embedding-3-small, W2-1b)
_TABLE_FOR_PROVIDER = {
    "offline": "food_embeddings",
    "openai": "food_embeddings_oai",
}
_DIM_FOR_PROVIDER = {
    "offline": EMBEDDING_DIM,
    "openai": OPENAI_EMBEDDING_DIM,
}
_MODEL_ID_FOR_PROVIDER = {
    "offline": EMBEDDING_MODEL_ID,
    "openai": OPENAI_EMBEDDING_MODEL_ID,
}


@dataclass
class SemanticMatch:
    food_id: uuid.UUID
    name: str
    score: float  # 1.0 = perfect; higher = better
    source: str  # "semantic" | "lexical_fallback"


def _source_text_for(food: FoodEntry) -> str:
    parts: list[str] = [food.name or ""]
    if food.common_names:
        parts.extend(food.common_names)
    if food.category:
        parts.append(food.category)
    if food.subcategory:
        parts.append(food.subcategory)
    return " | ".join(p for p in parts if p)


def _vec_literal(vec: list[float]) -> str:
    """Format a Python list as a pgvector literal: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


async def rebuild_embeddings(
    db: AsyncSession, batch_size: int = 200, provider: str = "offline"
) -> int:
    """Compute embeddings for every food in food_database.

    W2-1b: provider-aware. "offline" writes to food_embeddings (384-dim),
    "openai" writes to food_embeddings_oai (1536-dim). Idempotent: wipes the
    selected sidecar table and rebuilds from scratch. Safe for test/working
    DB. DO NOT run against Railway prod without a gated foodsci-ingestion
    sprint.
    """
    if provider not in _TABLE_FOR_PROVIDER:
        raise ValueError(f"unknown provider: {provider}")
    table = _TABLE_FOR_PROVIDER[provider]
    dim = _DIM_FOR_PROVIDER[provider]
    model_id = _MODEL_ID_FOR_PROVIDER[provider]

    await db.execute(text(f"TRUNCATE TABLE {table}"))
    result = await db.execute(select(FoodEntry))
    foods = list(result.scalars().all())

    count = 0
    for food in foods:
        src = _source_text_for(food)
        vec = embed_text(src, dim=dim, provider=provider)
        # Safety: the openai fallback path returns a 384-dim offline vector
        # on API failure. We refuse to mix spaces inside a single index.
        if len(vec) != dim:
            raise RuntimeError(
                f"embedding dim mismatch for provider={provider}: got {len(vec)}, expected {dim} "
                "(likely provider fallback triggered — rerun with the target provider healthy)"
            )
        await db.execute(
            text(
                f"INSERT INTO {table} (food_id, embedding, model, model_version, source_text) "
                "VALUES (:fid, (:vec)::vector, :model, :mver, :src)"
            ),
            {
                "fid": food.id,
                "vec": _vec_literal(vec),
                "model": model_id,
                "mver": MODEL_VERSION,
                "src": src,
            },
        )
        count += 1
    await db.commit()
    logger.info("Rebuilt %d food embeddings (provider=%s, model=%s)", count, provider, model_id)
    return count


async def semantic_search(
    db: AsyncSession,
    query: str,
    limit: int = 10,
    min_score: float = 0.0,
    provider: str = "offline",
) -> list[SemanticMatch]:
    """Return top-k semantic matches for a free-text query.

    W2-1b: provider-aware. provider="openai" targets the 1536-dim sidecar
    (food_embeddings_oai) and calls the live OpenAI embedder for the query;
    on API failure the outer embed_text returns an offline 384-dim vector —
    we detect the mismatch and fall back to lexical search rather than
    corrupting pgvector with a wrong-space vector.

    Falls back to lexical search if the embeddings table is empty.
    """
    if provider not in _TABLE_FOR_PROVIDER:
        raise ValueError(f"unknown provider: {provider}")
    table = _TABLE_FOR_PROVIDER[provider]
    dim = _DIM_FOR_PROVIDER[provider]

    async def _lexical() -> list[SemanticMatch]:
        foods, _ = await food_service.search_foods(db, query, limit=limit)
        return [
            SemanticMatch(food_id=f.id, name=f.name, score=0.5, source="lexical_fallback")
            for f in foods
        ]

    # Cold-start / undeployed-pgvector safety: if the sidecar table is empty,
    # missing (migrations not run), or the vector extension is unavailable, fall
    # back to lexical search rather than dead-ending the API.
    try:
        has_embeddings = (await db.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))).first()
    except SQLAlchemyError:
        await db.rollback()
        return await _lexical()
    if not has_embeddings:
        return await _lexical()

    qvec = embed_text(query, dim=dim, provider=provider)
    if len(qvec) != dim:
        logger.warning(
            "embed_text fell back to offline space for provider=%s; using lexical search",
            provider,
        )
        return await _lexical()
    qlit = _vec_literal(qvec)
    sql = text(
        f"""
        SELECT fe.food_id, fd.name, 1 - (fe.embedding <=> ('{qlit}')::vector) AS score
        FROM {table} fe
        JOIN food_database fd ON fd.id = fe.food_id
        ORDER BY fe.embedding <=> ('{qlit}')::vector
        LIMIT :lim
        """
    )
    try:
        rows = (await db.execute(sql, {"lim": limit})).all()
    except SQLAlchemyError:
        await db.rollback()
        return await _lexical()
    return [
        SemanticMatch(food_id=r[0], name=r[1], score=float(r[2]), source="semantic")
        for r in rows
        if float(r[2]) >= min_score
    ]
