#!/usr/bin/env python3
"""CLI script to ingest the allergen knowledge base into the database.

Usage:
    python -m scripts.ingest_food_data [--json-path PATH]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import async_session_factory  # noqa: E402
from app.services.food_ingestion import ingest_allergen_knowledge_base  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main(json_path: str | None = None) -> None:
    async with async_session_factory() as session:
        try:
            count = await ingest_allergen_knowledge_base(session, json_path)
            await session.commit()
            logger.info("Successfully ingested %d foods.", count)
        except Exception:
            await session.rollback()
            logger.exception("Ingestion failed")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest allergen knowledge base")
    parser.add_argument("--json-path", help="Path to allergen_knowledge_base_complete.json")
    args = parser.parse_args()
    asyncio.run(main(args.json_path))
