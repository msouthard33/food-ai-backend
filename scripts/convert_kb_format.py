#!/usr/bin/env python3
"""
convert_kb_format.py — Sync the master allergen knowledge base to the backend data directory.

WHY THIS SCRIPT EXISTS
----------------------
The backend service (food_ingestion.py) reads the master KB natively from:
  /Food AI App/04 - Food Science & Data/allergen_knowledge_base_complete.json

When deployed via Docker, the app falls back to:
  /app/data/allergen_knowledge_base_complete.json

This script validates, enriches, and copies the master KB to:
  /Food AI App/backend/data/allergen_knowledge_base_complete.json
  (which is bind-mounted into Docker as /app/data/)

ALLERGEN KEY → ComponentType ENUM MAPPING
------------------------------------------
The following table documents how every allergen_profile key in the KB
maps to the ComponentType enum values defined in app/models/enums.py.

  KB key               → ComponentType enum value
  ─────────────────────────────────────────────────
  gluten               → GLUTEN
  dairy                → MILK_DAIRY
  soy                  → SOY
  egg                  → EGGS
  tree_nuts            → TREE_NUTS
  peanuts              → PEANUTS
  fish                 → FISH
  shellfish            → SHELLFISH
  histamine            → HISTAMINES
  salicylates          → SALICYLATES
  oxalates             → OXALATES
  amines               → AMINES
  sulfites             → SULFITES
  nickel               → NICKEL
  fodmap_fructans      → FODMAP   (max wins when multiple FODMAP keys map to same type)
  fodmap_gos           → FODMAP
  fodmap_lactose       → LACTOSE
  fodmap_fructose      → FRUCTOSE
  fodmap_polyols       → FODMAP
  lectins              → LECTINS
  bromelain            → BROMELAIN   (future / specialty foods)
  nitrates             → NITRATES    (future / cured meats)

SCORE SCALE
-----------
The KB uses a 0–100 integer scale:
  0        = none / absent
  1–20     = very_low
  21–40    = low
  41–60    = moderate
  61–80    = high
  81–100   = very_high / extreme

The food_ingestion.py service stores raw 0–100 scores directly in
FoodComponentDetail.level_score (Decimal). No normalization is applied.

Usage:
    python backend/scripts/convert_kb_format.py [--dry-run] [--verbose]
    python backend/scripts/convert_kb_format.py --src /custom/path/kb.json --dst /custom/out.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import date

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (resolved relative to this script's location)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent          # backend/scripts/
BACKEND_DIR = SCRIPT_DIR.parent             # backend/
PROJECT_ROOT = BACKEND_DIR.parent           # /Food AI App/

DEFAULT_SRC = PROJECT_ROOT / "04 - Food Science & Data" / "allergen_knowledge_base_complete.json"
DEFAULT_DST = BACKEND_DIR / "data" / "allergen_knowledge_base_complete.json"

# ---------------------------------------------------------------------------
# Allergen key → ComponentType mapping (mirrors food_ingestion.py exactly)
# ---------------------------------------------------------------------------
ALLERGEN_KEY_MAP: dict[str, str] = {
    "gluten":           "gluten",
    "dairy":            "milk_dairy",
    "soy":              "soy",
    "egg":              "eggs",
    "tree_nuts":        "tree_nuts",
    "peanuts":          "peanuts",
    "fish":             "fish",
    "shellfish":        "shellfish",
    "histamine":        "histamines",
    "salicylates":      "salicylates",
    "oxalates":         "oxalates",
    "amines":           "amines",
    "sulfites":         "sulfites",
    "nickel":           "nickel",
    "fodmap_fructans":  "fodmap",
    "fodmap_gos":       "fodmap",
    "fodmap_lactose":   "lactose",
    "fodmap_fructose":  "fructose",
    "fodmap_polyols":   "fodmap",
    "lectins":          "lectins",
    "bromelain":        "bromelain",
    "nitrates":         "nitrates",
}

# Required top-level allergen keys every food entry should have
REQUIRED_ALLERGEN_KEYS = {
    "gluten", "dairy", "soy", "egg", "tree_nuts", "peanuts",
    "fish", "shellfish", "nickel", "histamine",
    "fodmap_fructans", "fodmap_gos", "fodmap_lactose", "fodmap_fructose", "fodmap_polyols",
    "salicylates", "oxalates", "lectins", "sulfites", "amines",
}

VALID_LEVELS = {"none", "very_low", "low", "low_moderate", "moderate", "moderate_high", "high", "very_high", "extreme"}


def validate_food_entry(food: dict, idx: int, verbose: bool = False) -> list[str]:
    """Return list of validation warnings for a single food entry."""
    warnings = []
    name = food.get("name", f"<entry {idx}>")

    if not food.get("id"):
        warnings.append(f"[{name}] Missing 'id'")
    if not food.get("name"):
        warnings.append(f"[entry {idx}] Missing 'name'")
    if not food.get("category"):
        warnings.append(f"[{name}] Missing 'category'")

    profile = food.get("allergen_profile", {})
    if not profile:
        warnings.append(f"[{name}] Empty allergen_profile")
        return warnings

    for key in REQUIRED_ALLERGEN_KEYS:
        if key not in profile:
            warnings.append(f"[{name}] Missing allergen key: {key}")

    for key, val in profile.items():
        if key not in ALLERGEN_KEY_MAP and verbose:
            warnings.append(f"[{name}] Unknown allergen key '{key}' (no ComponentType mapping)")
        score = val.get("score")
        if score is None:
            warnings.append(f"[{name}] allergen '{key}' missing 'score'")
        elif not (0 <= score <= 100):
            warnings.append(f"[{name}] allergen '{key}' score {score} out of 0–100 range")
        level = val.get("level", "")
        if level and level not in VALID_LEVELS:
            warnings.append(f"[{name}] allergen '{key}' has unknown level '{level}'")

    return warnings


def convert(src: Path, dst: Path, dry_run: bool = False, verbose: bool = False) -> int:
    """Read master KB, validate, and write to backend data directory.

    Returns count of food entries written.
    """
    logger.info("Source: %s", src)
    logger.info("Destination: %s", dst)

    if not src.exists():
        logger.error("Source file not found: %s", src)
        sys.exit(1)

    with open(src, encoding="utf-8") as f:
        kb = json.load(f)

    foods = kb.get("foods", [])
    logger.info("Loaded %d foods from master KB (version %s)", len(foods), kb.get("version", "?"))

    # Validate all entries
    all_warnings: list[str] = []
    names_seen: set[str] = set()
    duplicates: list[str] = []

    for idx, food in enumerate(foods):
        warnings = validate_food_entry(food, idx, verbose=verbose)
        all_warnings.extend(warnings)
        name = food.get("name", "")
        if name in names_seen:
            duplicates.append(name)
        names_seen.add(name)

    if duplicates:
        logger.warning("DUPLICATE food names detected: %s", duplicates)
    if all_warnings:
        for w in all_warnings:
            logger.warning(w)
        logger.warning("%d validation warnings across %d foods", len(all_warnings), len(foods))
    else:
        logger.info("Validation PASSED — no warnings")

    # Update the metadata to match reality
    output_kb = {
        "version": kb.get("version", "unknown"),
        "last_updated": str(date.today()),
        "total_items": len(foods),
        "allergen_key_map": ALLERGEN_KEY_MAP,
        "foods": foods,
    }

    logger.info("Writing %d foods to destination...", len(foods))

    if dry_run:
        logger.info("[DRY RUN] Would write %d bytes to %s", len(json.dumps(output_kb)), dst)
        logger.info("[DRY RUN] Skipping file write.")
        return len(foods)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(output_kb, f, indent=2, ensure_ascii=False)

    actual_size = dst.stat().st_size
    logger.info("Written: %s (%.1f KB)", dst, actual_size / 1024)
    logger.info("Total foods in backend/data/: %d", len(foods))
    return len(foods)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync master allergen KB → backend/data/ for Docker deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="Path to master KB JSON")
    parser.add_argument("--dst", default=str(DEFAULT_DST), help="Path to write backend KB JSON")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, do not write")
    parser.add_argument("--verbose", action="store_true", help="Show all warnings including unknown keys")
    args = parser.parse_args()

    count = convert(Path(args.src), Path(args.dst), dry_run=args.dry_run, verbose=args.verbose)

    print(f"\n✓ convert_kb_format.py complete — {count} foods synced to backend/data/")
    print(f"  Source: {args.src}")
    print(f"  Dest:   {args.dst}")
    if args.dry_run:
        print("  [DRY RUN — no file written]")


if __name__ == "__main__":
    main()
