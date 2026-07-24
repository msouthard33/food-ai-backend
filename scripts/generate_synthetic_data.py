"""CLI script to generate synthetic patient cohort for cold-start population prior.

Usage:
    python -m scripts.generate_synthetic_data [--n-patients N] [--dry-run]

Options:
    --n-patients N   Number of synthetic patients to generate (default: 150)
    --dry-run        Generate profiles and diaries in memory, print stats, do NOT insert to DB
    --kb-path PATH   Path to allergen_knowledge_base_complete.json
                     (default: auto-detected relative to this script)

Examples:
    python -m scripts.generate_synthetic_data --dry-run
    python -m scripts.generate_synthetic_data --n-patients 10 --dry-run
    python -m scripts.generate_synthetic_data --n-patients 150
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime, timedelta, timezone

# Ensure the backend app is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default KB path — relative to this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_KB_PATH = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "../../04 - Food Science & Data/allergen_knowledge_base_complete.json")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient cohort for Food AI cold-start.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=150,
        metavar="N",
        help="Number of synthetic patients to generate (default: 150)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate in memory only — do not insert to DB",
    )
    parser.add_argument(
        "--kb-path",
        default=_DEFAULT_KB_PATH,
        metavar="PATH",
        help=f"Path to KB JSON (default: {_DEFAULT_KB_PATH})",
    )
    return parser.parse_args()


def run_dry(n_patients: int, kb_path: str) -> None:
    """Generate profiles and diaries in memory and print stats. No DB interaction."""
    from app.services.synthetic_data_generator import (
        COMPONENT_STR_TO_ENUM,
        SYNTHETIC_CONDITION_MIX,
        generate_patient_diary,
        generate_patient_profile,
        load_kb_food_index,
    )

    logger.info("DRY RUN — loading KB from %s", kb_path)
    kb_index, safe_foods = load_kb_food_index(kb_path)
    logger.info(
        "KB: %d trigger buckets, %d safe foods",
        len(kb_index), len(safe_foods),
    )

    rng = random.Random(42)
    condition_options = [c for c, _ in SYNTHETIC_CONDITION_MIX]
    condition_weights = [w for _, w in SYNTHETIC_CONDITION_MIX]
    start_date = datetime.now(timezone.utc) - timedelta(weeks=8)

    total_meals = 0
    total_logged_meals = 0
    total_symptoms = 0
    condition_counts: dict[str, int] = {}

    for i in range(n_patients):
        primary = rng.choices(condition_options, weights=condition_weights, k=1)[0]
        conditions = [primary]
        if rng.random() < 0.30:
            others = [c for c in condition_options if c != primary]
            conditions.append(rng.choice(others))

        seed = rng.randint(0, 2**31 - 1)
        profile = generate_patient_profile(conditions, i, seed, kb_index, safe_foods)
        diary = generate_patient_diary(profile, kb_index, start_date)

        n_logged = sum(1 for m in diary["meals"] if m["logged"])
        total_meals += len(diary["meals"])
        total_logged_meals += n_logged
        total_symptoms += len(diary["symptoms"])
        condition_counts[primary] = condition_counts.get(primary, 0) + 1

    print("\n" + "=" * 60)
    print(f"DRY RUN STATS — {n_patients} synthetic patients")
    print("=" * 60)
    print(f"  Total meal events:         {total_meals}")
    print(f"  Total logged meals:        {total_logged_meals}")
    print(f"  Total symptom events:      {total_symptoms}")
    print(f"  Avg logged meals/patient:  {total_logged_meals / max(n_patients, 1):.1f}")
    print(f"  Avg symptoms/patient:      {total_symptoms / max(n_patients, 1):.1f}")
    print(f"  Logging consistency:       {total_logged_meals / max(total_meals, 1) * 100:.1f}%")
    print("\n  Condition distribution:")
    for cond, count in sorted(condition_counts.items(), key=lambda x: -x[1]):
        print(f"    {cond:<25}  {count:3d}  ({count / n_patients * 100:.0f}%)")
    print("=" * 60)
    print("No data was written to the database.")


async def run_insert(n_patients: int, kb_path: str) -> None:
    """Generate and insert synthetic cohort into the database."""
    from app.database import async_session_factory
    from app.services.synthetic_data_generator import generate_synthetic_cohort

    logger.info("Generating %d synthetic patients and inserting to DB...", n_patients)
    async with async_session_factory() as session:
        summary = await generate_synthetic_cohort(session, kb_path, n_patients=n_patients)

    print("\n" + "=" * 60)
    print(f"COHORT GENERATION COMPLETE — {summary['n_patients']} patients")
    print("=" * 60)
    print(f"  Total logged meals:        {summary['total_logged_meals']}")
    print(f"  Total symptom events:      {summary['total_symptoms']}")
    print(f"  Avg logged meals/patient:  {summary['avg_meals_per_patient']}")
    print(f"  Avg symptoms/patient:      {summary['avg_symptoms_per_patient']}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.kb_path):
        logger.error("KB file not found: %s", args.kb_path)
        sys.exit(1)

    if args.dry_run:
        run_dry(args.n_patients, args.kb_path)
    else:
        asyncio.run(run_insert(args.n_patients, args.kb_path))


if __name__ == "__main__":
    main()
