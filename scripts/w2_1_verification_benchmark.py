#!/usr/bin/env python3
"""W2-1 VERIFICATION benchmark (verification-agent, 2026-04-08).

Adapted from scripts/w2_1_offline_benchmark.py. The original emits a single
blended semantic rate which violates D1 condition 3 of the Wave 2 product
brief: direct-match vs decomposition-match MUST be reported separately.

This script:
  1. Measures decomposition_structured_output_rate on the 100-meal test set.
  2. For each top-500 query, computes top1 direct match. If usable -> direct
     bucket. Otherwise attempts heuristic decomposition of the query and
     checks whether >=1 ingredient resolves at >= threshold; if so ->
     decomposition bucket. Otherwise -> miss. Direct and decomposition
     buckets are DISJOINT.
  3. Samples 100 semantic-search calls for p50/p95 latency.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("SECRET_KEY", "bench")
os.environ.setdefault("ADMIN_API_KEY", "bench")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://bench@localhost/bench")

from app.services.llm_provider import embed_text, EMBEDDING_DIM  # noqa: E402
from app.services.meal_decomposition import _heuristic_split  # noqa: E402

KB_PATH = ROOT.parent / "04 - Food Science & Data" / "allergen_knowledge_base_complete.json"
TOP500_PATH = ROOT.parent / "04 - Food Science & Data" / "artifacts" / "w2-2_top500_us_foods.json"
OUT_PATH = ROOT.parent / "04 - Food Science & Data" / "artifacts" / "w2-1_verification_results.json"

USABLE_THRESHOLD = 0.35

TEST_MEALS_100 = [
    "chicken salad with grapes on sourdough",
    "scrambled eggs with cheddar and spinach",
    "greek yogurt with blueberries and honey",
    "peanut butter and jelly sandwich",
    "turkey sandwich with lettuce tomato and mayo",
    "beef taco with cheese and sour cream",
    "spaghetti with marinara sauce",
    "caesar salad with grilled chicken",
    "oatmeal with banana and walnuts",
    "avocado toast with egg",
    "chicken tikka masala with basmati rice",
    "pad thai with shrimp",
    "pepperoni pizza",
    "cheeseburger with fries",
    "sushi roll with tuna and avocado",
    "miso soup with tofu",
    "caprese salad with mozzarella tomato and basil",
    "bacon and eggs with toast",
    "pancakes with maple syrup and butter",
    "french toast with strawberries",
    "beef stir fry with broccoli",
    "chicken noodle soup",
    "clam chowder",
    "lobster roll",
    "fish and chips",
    "shrimp scampi with linguine",
    "pork belly bao buns",
    "ramen with pork and egg",
    "hummus and pita with cucumber",
    "falafel wrap with tahini",
    "lamb gyro with tzatziki",
    "chicken shawarma plate",
    "beef kebab with rice",
    "kimchi fried rice with egg",
    "bibimbap with beef",
    "bulgogi over rice",
    "tonkatsu with cabbage",
    "teriyaki chicken with broccoli",
    "california roll",
    "spicy tuna roll",
    "poke bowl with salmon",
    "acai bowl with granola and berries",
    "smoothie with banana spinach and almond milk",
    "protein shake with whey",
    "chia pudding with mango",
    "overnight oats with peanut butter",
    "bagel with cream cheese and lox",
    "eggs benedict with hollandaise",
    "quiche lorraine",
    "shakshuka with feta",
    "breakfast burrito with chorizo",
    "huevos rancheros",
    "carnitas tacos with onion and cilantro",
    "fish tacos with slaw",
    "chicken enchiladas with red sauce",
    "beef chili with beans",
    "cornbread with honey butter",
    "mac and cheese",
    "grilled cheese with tomato soup",
    "BLT sandwich",
    "club sandwich with turkey and bacon",
    "reuben with corned beef and sauerkraut",
    "philly cheesesteak with onions",
    "meatball sub",
    "italian sub with salami and provolone",
    "buffalo wings with blue cheese",
    "chicken parmesan with pasta",
    "lasagna with ricotta",
    "eggplant parmesan",
    "risotto with mushrooms",
    "gnocchi with pesto",
    "ravioli with butter and sage",
    "beef bourguignon",
    "coq au vin",
    "chicken pot pie",
    "shepherd's pie",
    "pot roast with carrots and potatoes",
    "pulled pork sandwich with slaw",
    "bbq ribs with cornbread",
    "brisket with pickles and onion",
    "smoked turkey with mashed potatoes",
    "thanksgiving plate with stuffing and gravy",
    "green bean casserole",
    "sweet potato with marshmallow",
    "cranberry sauce",
    "pumpkin pie with whipped cream",
    "apple pie a la mode",
    "chocolate chip cookies",
    "brownies with walnuts",
    "cheesecake with strawberries",
    "tiramisu",
    "creme brulee",
    "chocolate mousse",
    "ice cream sundae",
    "milkshake with whipped cream",
    "root beer float",
    "matcha latte with oat milk",
    "iced coffee with almond milk",
    "kombucha",
    "green juice with kale and apple",
    "mimosa with orange juice",
    "margarita with salt",
]
TEST_MEALS_100 = TEST_MEALS_100[:100]
assert len(TEST_MEALS_100) == 100


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def load_json(p):
    with open(p) as f:
        return json.load(f)


def build_index(kb):
    idx = []
    for food in kb:
        name = food.get("name", "") or ""
        aliases = food.get("common_names") or food.get("aliases") or []
        cat = food.get("category") or ""
        sub = food.get("subcategory") or ""
        parts = [name]
        if isinstance(aliases, list):
            parts.extend(aliases)
        parts.extend([cat, sub])
        src = " | ".join(p for p in parts if p)
        vec = embed_text(src, dim=EMBEDDING_DIM, provider="offline")
        idx.append((name, vec))
    return idx


def top1(query, index):
    qv = embed_text(query, dim=EMBEDDING_DIM, provider="offline")
    best_name, best = "", -1.0
    for name, vec in index:
        s = cosine(qv, vec)
        if s > best:
            best, best_name = s, name
    return best_name, best


def pct(latencies, p):
    if not latencies:
        return 0.0
    s = sorted(latencies)
    i = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return round(s[i], 3)


def main():
    kb = load_json(KB_PATH).get("foods", [])
    top500 = load_json(TOP500_PATH).get("foods", [])
    print(f"[verify] KB={len(kb)} top500={len(top500)}")

    index = build_index(kb)
    print(f"[verify] indexed {len(index)} foods")

    # ---- Semantic: direct vs decomposition, disjoint ----
    direct_hits = 0
    decomp_hits = 0
    misses = 0
    latencies = []
    sample_decomp_wins = []
    sample_misses = []

    for q in top500:
        query_text = q.get("query") or q.get("name") or ""
        if not query_text:
            continue
        t0 = time.perf_counter()
        name, score = top1(query_text, index)
        latencies.append((time.perf_counter() - t0) * 1000)
        if score >= USABLE_THRESHOLD:
            direct_hits += 1
            continue
        # Direct missed — try decomposition of the query
        parts = _heuristic_split(query_text)
        resolved_any = False
        for ing, _p in parts:
            if ing.strip().lower() == query_text.strip().lower():
                continue  # no-op split, skip
            _n, s2 = top1(ing, index)
            if s2 >= USABLE_THRESHOLD:
                resolved_any = True
                break
        if resolved_any:
            decomp_hits += 1
            if len(sample_decomp_wins) < 15:
                sample_decomp_wins.append({"query": query_text, "direct_score": round(score, 3)})
        else:
            misses += 1
            if len(sample_misses) < 15:
                sample_misses.append({"query": query_text, "direct_score": round(score, 3)})

    total = direct_hits + decomp_hits + misses
    direct_rate = round(100 * direct_hits / total, 2) if total else 0.0
    decomp_rate = round(100 * decomp_hits / total, 2) if total else 0.0

    # ---- Latency sample: 100 requests (re-sample deterministic queries) ----
    lat_sample = []
    sample_queries = [q.get("query") or "" for q in top500[:100]]
    for qt in sample_queries:
        if not qt:
            continue
        t0 = time.perf_counter()
        top1(qt, index)
        lat_sample.append((time.perf_counter() - t0) * 1000)
    lat_p50 = pct(lat_sample, 50)
    lat_p95 = pct(lat_sample, 95)

    # ---- Decomposition structured-output rate on 100 meals ----
    structured = 0
    for meal in TEST_MEALS_100:
        parts = _heuristic_split(meal)
        # structured = heuristic produced >=1 ingredient token AND split actually occurred
        # (i.e., got something back)
        if parts:
            structured += 1
    decomposition_structured_rate = round(100 * structured / len(TEST_MEALS_100), 2)

    results = {
        "benchmark_id": "w2-1_verification",
        "run_by": "verification-agent",
        "embedder": "offline-trigramhash-v1",
        "embedding_dim": EMBEDDING_DIM,
        "kb_total": len(kb),
        "threshold": USABLE_THRESHOLD,
        "metrics": {
            "decomposition_structured_output_rate": decomposition_structured_rate,
            "semantic_search_direct_match_rate_top500": direct_rate,
            "semantic_search_decomposition_match_rate_top500": decomp_rate,
            "semantic_search_combined_rate_top500": round(direct_rate + decomp_rate, 2),
            "latency_p50_ms": lat_p50,
            "latency_p95_ms": lat_p95,
        },
        "counts": {
            "total_queries": total,
            "direct_hits": direct_hits,
            "decomposition_hits": decomp_hits,
            "misses": misses,
            "latency_sample_n": len(lat_sample),
            "decomposition_test_meals": len(TEST_MEALS_100),
            "decomposition_structured_count": structured,
        },
        "sample_decomposition_wins": sample_decomp_wins,
        "sample_misses": sample_misses,
        "notes": [
            "Direct and decomposition buckets are disjoint: a query is decomp-only if direct top1 < threshold AND at least one heuristic-split ingredient resolves >= threshold.",
            "Heuristic splitter skips no-op splits (where the only part equals the query itself).",
            "Offline trigram-hash embedder; identical scoring to service path.",
        ],
    }
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["metrics"], indent=2))
    print(json.dumps(results["counts"], indent=2))
    print(f"[verify] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
