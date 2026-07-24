#!/usr/bin/env python3
"""W2-1 offline acceptance benchmark.

Measures the three W2-1 metrics without needing a live DB or live LLM:
  - decomposition_structured_rate (heuristic path; LLM unavailable here)
  - semantic_match_rate_top500 (vs. artifacts/w2-2_top500_us_foods.json)
  - latency_p50_ms / latency_p95_ms

Runs in-process against allergen_knowledge_base_complete.json (v2.3.0, 546
foods). This mirrors what pgvector would return — the trigram-hash embedder
is deterministic and the scoring function is identical (cosine similarity).

Writes a JSON results file the aiml-sprint report links as an artifact.

Usage:
    cd backend && .venv/bin/python scripts/w2_1_offline_benchmark.py

Exits 0 on completion. Does NOT gate on thresholds — the aiml-sprint report
interprets the numbers and decides status.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

# Make `app` importable so we reuse the real embed_text / heuristic splitter
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Stub out settings so importing llm_provider doesn't require env vars
import os
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("SECRET_KEY", "bench")
os.environ.setdefault("ADMIN_API_KEY", "bench")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://bench@localhost/bench")

from app.services.llm_provider import embed_text, EMBEDDING_DIM  # noqa: E402
from app.services.meal_decomposition import _heuristic_split  # noqa: E402

KB_PATH = ROOT.parent / "04 - Food Science & Data" / "allergen_knowledge_base_complete.json"
TOP500_PATH = ROOT.parent / "04 - Food Science & Data" / "artifacts" / "w2-2_top500_us_foods.json"
OUT_PATH = (
    ROOT.parent
    / "04 - Food Science & Data"
    / "artifacts"
    / "w2-1_benchmark_results.json"
)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    # vectors are L2-normalized in embed_text, so dot == cosine
    return dot


def load_kb() -> list[dict]:
    with open(KB_PATH) as f:
        data = json.load(f)
    return data.get("foods", [])


def load_top500() -> list[dict]:
    if not TOP500_PATH.exists():
        raise SystemExit(f"FAIL-FAST: top500 artifact missing at {TOP500_PATH}")
    with open(TOP500_PATH) as f:
        data = json.load(f)
    foods = data.get("foods")
    if not isinstance(foods, list):
        raise SystemExit("FAIL-FAST: top500 artifact schema invalid (missing 'foods' list)")
    return foods


def build_kb_index(kb: list[dict]) -> list[tuple[str, list[float], dict]]:
    idx: list[tuple[str, list[float], dict]] = []
    for food in kb:
        name = food.get("name", "") or ""
        aliases = food.get("common_names") or food.get("aliases") or []
        cat = food.get("category") or ""
        sub = food.get("subcategory") or ""
        src = " | ".join(
            p for p in [name, *([] if not isinstance(aliases, list) else aliases), cat, sub] if p
        )
        vec = embed_text(src, dim=EMBEDDING_DIM, provider="offline")
        idx.append((name, vec, food))
    return idx


def top1(query: str, index: list[tuple[str, list[float], dict]]) -> tuple[str, float]:
    qv = embed_text(query, dim=EMBEDDING_DIM, provider="offline")
    best_name = ""
    best_score = -1.0
    for name, vec, _ in index:
        s = cosine(qv, vec)
        if s > best_score:
            best_score = s
            best_name = name
    return best_name, best_score


# ---------- Semantic match benchmark ----------

USABLE_THRESHOLD = 0.35  # cosine score threshold; empirically chosen for trigram-hash


def semantic_benchmark(index, queries: list[dict]) -> dict:
    latencies: list[float] = []
    usable = 0
    total = 0
    hits: list[dict] = []
    for q in queries:
        query_text = q.get("query") or q.get("name") or ""
        if not query_text:
            continue
        total += 1
        t0 = time.perf_counter()
        name, score = top1(query_text, index)
        latencies.append((time.perf_counter() - t0) * 1000)
        is_usable = score >= USABLE_THRESHOLD
        if is_usable:
            usable += 1
        hits.append(
            {"query": query_text, "match": name, "score": round(score, 3), "usable": is_usable}
        )
    latencies.sort()
    def pct(p):
        if not latencies:
            return 0.0
        i = max(0, min(len(latencies) - 1, int(round(p / 100 * (len(latencies) - 1)))))
        return round(latencies[i], 2)
    return {
        "total": total,
        "usable": usable,
        "rate_pct": round(100 * usable / total, 2) if total else 0.0,
        "threshold": USABLE_THRESHOLD,
        "latency_p50_ms": pct(50),
        "latency_p95_ms": pct(95),
        "sample_hits": hits[:20],
        "sample_misses": [h for h in hits if not h["usable"]][:20],
    }


# ---------- Decomposition benchmark ----------

# A 100-meal synthetic test set. Heuristic-only (no LLM available in offline run).
# Structured output = heuristic produced >=1 ingredient token.
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
assert len(TEST_MEALS_100) == 100, f"test set must be 100 meals, got {len(TEST_MEALS_100)}"


def decomposition_benchmark(index) -> dict:
    structured_count = 0
    total_ingredients = 0
    total_resolved = 0
    details: list[dict] = []
    for meal in TEST_MEALS_100:
        parts = _heuristic_split(meal)
        if parts:
            structured_count += 1
        resolved_here = 0
        for ingredient, _portion in parts:
            total_ingredients += 1
            _name, score = top1(ingredient, index)
            if score >= USABLE_THRESHOLD:
                resolved_here += 1
                total_resolved += 1
        details.append({"meal": meal, "n_ingredients": len(parts), "n_resolved": resolved_here})
    return {
        "total_meals": len(TEST_MEALS_100),
        "structured_rate_pct": round(100 * structured_count / len(TEST_MEALS_100), 2),
        "total_ingredients": total_ingredients,
        "resolved_ingredients": total_resolved,
        "per_ingredient_kb_hit_rate_pct": (
            round(100 * total_resolved / total_ingredients, 2) if total_ingredients else 0.0
        ),
        "sample": details[:10],
    }


def main() -> int:
    print(f"[bench] loading KB from {KB_PATH}")
    kb = load_kb()
    print(f"[bench] KB foods: {len(kb)}")
    print(f"[bench] loading top500 from {TOP500_PATH}")
    top500 = load_top500()
    print(f"[bench] top500 queries: {len(top500)}")

    print("[bench] building in-memory embedding index...")
    t0 = time.perf_counter()
    index = build_kb_index(kb)
    build_ms = int((time.perf_counter() - t0) * 1000)
    print(f"[bench] indexed {len(index)} foods in {build_ms} ms")

    print("[bench] running semantic match benchmark...")
    sem = semantic_benchmark(index, top500)
    print(
        f"[bench] semantic: {sem['usable']}/{sem['total']} = {sem['rate_pct']}% "
        f"(p50 {sem['latency_p50_ms']} ms, p95 {sem['latency_p95_ms']} ms)"
    )

    print("[bench] running decomposition benchmark...")
    dec = decomposition_benchmark(index)
    print(
        f"[bench] decomposition: structured_rate={dec['structured_rate_pct']}% "
        f"per_ing_hit={dec['per_ingredient_kb_hit_rate_pct']}%"
    )

    results = {
        "benchmark_id": "w2-1_offline",
        "embedder": "offline-trigramhash-v1",
        "embedding_dim": EMBEDDING_DIM,
        "kb_total": len(kb),
        "kb_version": "2.3.0 (working)",
        "top500_total": len(top500),
        "index_build_ms": build_ms,
        "semantic_search": sem,
        "decomposition": dec,
        "notes": [
            "Offline run — no live LLM; decomposition uses heuristic splitter only.",
            "Live LLM run required to hit W2-1 90% structured-output target; wire OpenAI/Anthropic key and re-run decompose_meal_text with use_llm=True.",
            "Semantic scores come from trigram-hash cosine; same function is used by pgvector in the service path so match ordering is identical.",
        ],
    }
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
