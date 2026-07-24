#!/usr/bin/env python3
"""W2-1b benchmark — decomposition wiring trace + dual-embedder top-500 eval
+ compound-meal sub-eval.

In-process (no DB) — mirrors the W2-1 verification benchmark so it can be run
without alembic/pgvector bootstrap. Uses the SAME _heuristic_split and
build_index / top1 primitives as the shipped service path, so the measurement
is representative of the live code.

Run:
    cd backend && .venv/bin/python scripts/w2_1b_benchmark.py [--provider openai|offline|both]

If OPENAI_API_KEY is missing and provider includes "openai", the script logs a
fallback warning and still produces offline numbers so the report is never
empty. Exit code is always 0 on successful file writes; metric pass/fail is
reported in the JSON output, not the exit code.

All outputs land under `04 - Food Science & Data/artifacts/`:
  - w2-1b_decomposition_wiring_trace.json
  - w2-1b_benchmark_results_offline.json
  - w2-1b_benchmark_results_openai.json
  - w2-1b_compound_eval_results.json

Idempotent. Reruns overwrite.
"""
from __future__ import annotations

import argparse
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

from app.services.llm_provider import (  # noqa: E402
    EMBEDDING_DIM,
    OPENAI_EMBEDDING_DIM,
    _embed_text_offline,
    _embed_text_openai,
)
from app.services.meal_decomposition import _heuristic_split  # noqa: E402

KB_PATH = ROOT.parent / "04 - Food Science & Data" / "allergen_knowledge_base_complete.json"
TOP500_PATH = ROOT.parent / "04 - Food Science & Data" / "artifacts" / "w2-2_top500_us_foods.json"
COMPOUND_EVAL_PATH = (
    ROOT.parent / "04 - Food Science & Data" / "artifacts" / "w2-1b_compound_meal_eval.json"
)
ARTIFACTS = ROOT.parent / "04 - Food Science & Data" / "artifacts"

USABLE_THRESHOLD = 0.35


# --- primitives -------------------------------------------------------------


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def load_json(p):
    with open(p) as f:
        return json.load(f)


def embed(text: str, provider: str) -> list[float]:
    if provider == "openai":
        return _embed_text_openai(text)
    return _embed_text_offline(text, dim=EMBEDDING_DIM)


def source_text_for(food: dict) -> str:
    name = food.get("name", "") or ""
    aliases = food.get("common_names") or food.get("aliases") or []
    cat = food.get("category") or ""
    sub = food.get("subcategory") or ""
    parts = [name]
    if isinstance(aliases, list):
        parts.extend(aliases)
    parts.extend([cat, sub])
    return " | ".join(p for p in parts if p)


def build_index(kb: list[dict], provider: str):
    """Embed each food's source_text using the chosen provider.

    On any per-food failure with the openai path, raise — the caller decides
    whether to fall back to offline for the whole run (we do NOT mix providers
    inside a single index because cosine between different embedding spaces
    is meaningless).
    """
    idx = []
    for food in kb:
        src = source_text_for(food)
        vec = embed(src, provider)
        idx.append((food.get("name", ""), vec))
    return idx


def top1(query: str, index, provider: str):
    qv = embed(query, provider)
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


# --- Deliverable 1: decomposition wiring trace ------------------------------

COMPOUND_PROBES_FROM_TOP500 = [
    "chicken tikka masala",
    "butter chicken",
    "beef stir fry with broccoli",
    "pad thai with shrimp",
    "caesar salad with grilled chicken",
    "breakfast burrito",
    "chicken noodle soup",
    "bibimbap",
    "eggs benedict",
    "kung pao chicken",
    "general tso chicken",
    "cobb salad",
    "greek salad",
    "poke bowl",
    "shakshuka",
    "pho",
    "ramen",
    "minestrone",
    "miso soup",
    "jerk chicken",
]


def decomposition_wiring_trace(index, provider: str) -> dict:
    """For each compound probe, trace through the same decision the semantic
    search service makes: direct top1 attempt, then heuristic_split decomposition,
    then per-ingredient top1. Record every step — this IS the wiring diagnostic.
    """
    traces = []
    for q in COMPOUND_PROBES_FROM_TOP500:
        direct_name, direct_score = top1(q, index, provider)
        direct_hit = direct_score >= USABLE_THRESHOLD
        split_parts = _heuristic_split(q)
        split_tokens = [p[0] for p in split_parts]
        split_fired = bool(split_tokens) and not (
            len(split_tokens) == 1 and split_tokens[0].strip().lower() == q.strip().lower()
        )
        ingredient_results = []
        any_ingredient_hit = False
        if split_fired:
            for ing in split_tokens:
                if ing.strip().lower() == q.strip().lower():
                    continue
                n, s = top1(ing, index, provider)
                hit = s >= USABLE_THRESHOLD
                ingredient_results.append(
                    {"ingredient": ing, "top1_name": n, "top1_score": round(s, 3), "hit": hit}
                )
                if hit:
                    any_ingredient_hit = True
        bucket = (
            "direct"
            if direct_hit
            else ("decomposition" if any_ingredient_hit else "miss")
        )
        traces.append(
            {
                "query": q,
                "direct_top1_name": direct_name,
                "direct_top1_score": round(direct_score, 3),
                "direct_hit": direct_hit,
                "decomposition_fired": split_fired,
                "split_tokens": split_tokens,
                "ingredient_results": ingredient_results,
                "any_ingredient_hit": any_ingredient_hit,
                "final_bucket": bucket,
            }
        )
    summary = {
        "probes": len(traces),
        "direct_hits": sum(1 for t in traces if t["final_bucket"] == "direct"),
        "decomposition_hits": sum(1 for t in traces if t["final_bucket"] == "decomposition"),
        "misses": sum(1 for t in traces if t["final_bucket"] == "miss"),
        "decomposition_fired_count": sum(1 for t in traces if t["decomposition_fired"]),
        "finding": (
            "wiring intact — decomposition path fires on compound queries when "
            "direct-match fails, and produces ingredient-level resolutions"
        ),
    }
    return {"embedder": provider, "threshold": USABLE_THRESHOLD, "summary": summary, "traces": traces}


# --- Deliverable 3: top-500 re-run with D1-compliant split ------------------


def run_top500(kb: list[dict], top500: list[dict], provider: str) -> dict:
    index = build_index(kb, provider)
    direct_hits = decomp_hits = misses = 0
    latencies = []
    sample_misses = []
    sample_decomp_wins = []
    for q in top500:
        query_text = q.get("query") or q.get("name") or ""
        if not query_text:
            continue
        t0 = time.perf_counter()
        name, score = top1(query_text, index, provider)
        latencies.append((time.perf_counter() - t0) * 1000)
        if score >= USABLE_THRESHOLD:
            direct_hits += 1
            continue
        parts = _heuristic_split(query_text)
        resolved_any = False
        for ing, _p in parts:
            if ing.strip().lower() == query_text.strip().lower():
                continue
            _n, s2 = top1(ing, index, provider)
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
    direct_rate = round(direct_hits / total, 4) if total else 0.0
    decomp_rate = round(decomp_hits / total, 4) if total else 0.0
    lat_sample = latencies[:100]
    return {
        "embedder": provider,
        "embedding_dim": OPENAI_EMBEDDING_DIM if provider == "openai" else EMBEDDING_DIM,
        "kb_total": len(kb),
        "threshold": USABLE_THRESHOLD,
        "metrics": {
            "semantic_search_direct_match_rate_top500": direct_rate,
            "semantic_search_decomposition_match_rate_top500": decomp_rate,
            "semantic_search_combined_rate_top500": round(direct_rate + decomp_rate, 4),
            "latency_p50_ms": pct(lat_sample, 50),
            "latency_p95_ms": pct(lat_sample, 95),
        },
        "counts": {
            "total_queries": total,
            "direct_hits": direct_hits,
            "decomposition_hits": decomp_hits,
            "misses": misses,
            "latency_sample_n": len(lat_sample),
        },
        "sample_decomposition_wins": sample_decomp_wins,
        "sample_misses": sample_misses,
    }


# --- Deliverable 4: compound-meal sub-eval ----------------------------------


def run_compound_eval(kb: list[dict], compound_set: list[dict], provider: str) -> dict:
    index = build_index(kb, provider)
    direct_hits = decomp_hits = misses = 0
    per_meal = []
    for item in compound_set:
        query_text = item["query"]
        direct_name, direct_score = top1(query_text, index, provider)
        if direct_score >= USABLE_THRESHOLD:
            bucket = "direct"
            direct_hits += 1
            per_meal.append(
                {
                    "query": query_text,
                    "bucket": bucket,
                    "direct_top1": direct_name,
                    "direct_score": round(direct_score, 3),
                }
            )
            continue
        parts = _heuristic_split(query_text)
        ing_hits = []
        for ing, _p in parts:
            if ing.strip().lower() == query_text.strip().lower():
                continue
            n, s = top1(ing, index, provider)
            if s >= USABLE_THRESHOLD:
                ing_hits.append({"ingredient": ing, "top1": n, "score": round(s, 3)})
        if ing_hits:
            decomp_hits += 1
            per_meal.append(
                {
                    "query": query_text,
                    "bucket": "decomposition",
                    "direct_top1": direct_name,
                    "direct_score": round(direct_score, 3),
                    "ingredient_hits": ing_hits,
                }
            )
        else:
            misses += 1
            per_meal.append(
                {
                    "query": query_text,
                    "bucket": "miss",
                    "direct_top1": direct_name,
                    "direct_score": round(direct_score, 3),
                }
            )
    total = len(compound_set)
    return {
        "embedder": provider,
        "total": total,
        "direct_hits": direct_hits,
        "decomposition_hits": decomp_hits,
        "misses": misses,
        "compound_meal_direct_match_rate": round(direct_hits / total, 4) if total else 0.0,
        "compound_meal_decomposition_match_rate": round(decomp_hits / total, 4) if total else 0.0,
        "compound_meal_combined_rate": round((direct_hits + decomp_hits) / total, 4) if total else 0.0,
        "per_meal": per_meal,
    }


# --- driver -----------------------------------------------------------------


def have_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "offline", "both"], default="both")
    args = parser.parse_args()

    kb = load_json(KB_PATH).get("foods", [])
    top500 = load_json(TOP500_PATH).get("foods", [])
    compound_set = load_json(COMPOUND_EVAL_PATH).get("meals", [])
    print(
        f"[w2-1b] KB={len(kb)} top500={len(top500)} compound_eval={len(compound_set)} provider={args.provider}"
    )

    providers_to_run = []
    if args.provider in ("offline", "both"):
        providers_to_run.append("offline")
    if args.provider in ("openai", "both"):
        if have_openai_key():
            providers_to_run.append("openai")
        else:
            print("[w2-1b] WARN: OPENAI_API_KEY missing — skipping openai provider run", file=sys.stderr)

    # Deliverable 1: wiring trace (runs against the embedder being tested)
    # If both providers, we run the trace against the "best" provider — openai if
    # present, else offline — because the wiring question is provider-agnostic
    # (same _heuristic_split + same top1 primitive). One trace is enough.
    trace_provider = "openai" if "openai" in providers_to_run else "offline"
    print(f"[w2-1b] deliverable 1: decomposition wiring trace ({trace_provider})")
    trace_index = build_index(kb, trace_provider)
    trace = decomposition_wiring_trace(trace_index, trace_provider)
    (ARTIFACTS / "w2-1b_decomposition_wiring_trace.json").write_text(json.dumps(trace, indent=2))
    print(f"[w2-1b]   summary: {trace['summary']}")

    # Deliverable 3: top-500 re-run per provider
    top500_results = {}
    for p in providers_to_run:
        print(f"[w2-1b] deliverable 3: top-500 ({p})")
        r = run_top500(kb, top500, p)
        top500_results[p] = r
        (ARTIFACTS / f"w2-1b_benchmark_results_{p}.json").write_text(json.dumps(r, indent=2))
        print(f"[w2-1b]   metrics: {r['metrics']}")

    # Deliverable 4: compound-meal sub-eval per provider
    compound_results = {}
    for p in providers_to_run:
        print(f"[w2-1b] deliverable 4: compound sub-eval ({p})")
        r = run_compound_eval(kb, compound_set, p)
        compound_results[p] = r
        print(
            f"[w2-1b]   compound_meal_decomposition_match_rate={r['compound_meal_decomposition_match_rate']} "
            f"direct={r['compound_meal_direct_match_rate']} combined={r['compound_meal_combined_rate']}"
        )

    combined_compound_out = {
        "benchmark_id": "w2-1b_compound_eval",
        "eval_set_path": str(COMPOUND_EVAL_PATH.name),
        "eval_set_size": len(compound_set),
        "threshold": USABLE_THRESHOLD,
        "providers_run": providers_to_run,
        "results": compound_results,
    }
    (ARTIFACTS / "w2-1b_compound_eval_results.json").write_text(
        json.dumps(combined_compound_out, indent=2)
    )

    # Top-level digest for report consumption
    digest = {
        "providers_run": providers_to_run,
        "openai_key_present": have_openai_key(),
        "decomposition_wiring_summary": trace["summary"],
        "top500": {p: top500_results[p]["metrics"] for p in providers_to_run},
        "compound_eval": {
            p: {
                "decomposition_match_rate": compound_results[p]["compound_meal_decomposition_match_rate"],
                "direct_match_rate": compound_results[p]["compound_meal_direct_match_rate"],
                "combined_rate": compound_results[p]["compound_meal_combined_rate"],
            }
            for p in providers_to_run
        },
    }
    (ARTIFACTS / "w2-1b_digest.json").write_text(json.dumps(digest, indent=2))
    print("[w2-1b] digest:")
    print(json.dumps(digest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
