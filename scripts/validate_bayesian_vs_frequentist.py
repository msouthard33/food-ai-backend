"""Head-to-head validation: Beta-Binomial Bayesian engine vs frequentist proportion.

Bayesian Sprint 2 (Wave 2, Pillar 1). REPORT-ONLY — no engine constants are
edited, no endpoints change, nothing deploys. We ship the Bayesian model per the
Sprint-1 decision regardless; this harness gives Matt the honest, quantified
comparison and tunes the prior strength (kappa) / exposure threshold for Sprint 3.

WHAT IT DOES
------------
1. Builds a reproducible 150-patient synthetic cohort (generator seeded
   ``random.Random(42)``) in a local Postgres DB, capturing the in-memory
   ground-truth profile dicts as it goes (component-level truth per patient).
2. For every patient runs the real ``analyze_bayesian_triggers`` engine AND a
   frequentist component-level proportion baseline derived from the SAME 2x2 day
   counts the engine produces (so the ONLY difference is prior-vs-raw-proportion —
   a true apples-to-apples contrast).
3. Treats each ComponentType as a binary prediction of "is a true trigger for this
   patient" and computes precision / recall / F1 / ROC-AUC for both models, plus
   Bayesian calibration and a cold-start probe.
4. Sweeps a small kappa x exposure-threshold grid and recommends Sprint-3 defaults.
5. Prints a summary table and writes a JSON report conforming to the report schema.

DB SETUP (reproducibility)
--------------------------
The engine needs real ``FoodComponentDetail`` rows (KB join) + meals/symptoms, so
this runs against Postgres, not an in-process stub (the ORM is Postgres-specific:
UUID / JSONB / enum types). Point ``DATABASE_URL`` at a disposable local DB that
already carries the schema + KB. To build one from the test DB template:

    psql -h localhost -d postgres -c \
      "CREATE DATABASE foodai_bayes_val TEMPLATE foodai_test;"
    # truncate its data tables, then:
    DATABASE_URL=postgresql+asyncpg://<user>@localhost/foodai_bayes_val \
      python -m scripts.ingest_food_data --json-path data/allergen_knowledge_base_complete.json

Then:

    DATABASE_URL=postgresql+asyncpg://<user>@localhost/foodai_bayes_val \
      python -m scripts.validate_bayesian_vs_frequentist

The script deletes any pre-existing ``is_synthetic`` users at start so the
population-prior aggregate is built from exactly the cohort under test.

Pure Python only — AUC is computed by hand (rank-sum), no numpy/scipy.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# ── Env must be set BEFORE importing app (engine is built at import time) ──────
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://matthewsouthard@localhost/foodai_bayes_val",
)
os.environ.setdefault("SUPABASE_JWT_SECRET", "")
os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")

import logging  # noqa: E402
import random  # noqa: E402
import uuid  # noqa: E402

from sqlalchemy import delete, text  # noqa: E402

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

import app.models  # noqa: E402, F401 — register ORM models
from app.database import async_session_factory, engine  # noqa: E402

# app/database.py builds the engine with echo=True whenever APP_ENV != production,
# which floods stdout with raw SQL. Toggle it off for a readable validation run.
engine.echo = False
from app.models.enums import ConditionType  # noqa: E402
from app.models.user import User, UserCondition  # noqa: E402
from app.services import bayesian_trigger as bt  # noqa: E402
from app.services.bayesian_trigger import (  # noqa: E402
    analyze_bayesian_triggers,
    build_population_prior_table,
    prior_score,
)
from app.services.synthetic_data_generator import (  # noqa: E402
    COMPONENT_STR_TO_ENUM,
    SYNTHETIC_CONDITION_MIX,
    generate_patient_diary,
    generate_patient_profile,
    insert_synthetic_patient,
    load_kb_food_index,
)

logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_KB_PATH = str(_SCRIPT_DIR.parent / "data" / "allergen_knowledge_base_complete.json")
_DEFAULT_REPORT_PATH = str(
    _SCRIPT_DIR.parent.parent
    / "04 - Food Science & Data"
    / "reports"
    / "bayesian_validation_2026-07-24.json"
)

# Tuning grid (task-specified)
_KAPPA_GRID = [3.0, 6.0, 10.0]
_EXPOSURE_THRESHOLD_GRID = [1.5, 2.0, 2.5]
_DEFAULT_KAPPA = bt.PRIOR_STRENGTH_KAPPA          # 6.0
_DEFAULT_EXPOSURE = bt.EXPOSURE_LEVEL_THRESHOLD   # 2.0

_COHORT_SEED = 42
_N_PATIENTS = 150


# ── Pure-Python metric helpers ────────────────────────────────────────────────

def _roc_auc(pairs: list[tuple[float, int]]) -> float:
    """ROC-AUC via the rank-sum (Mann-Whitney U) identity, with tie-aware average
    ranks. ``pairs`` = list of (score, label in {0,1}). Returns 0.5 when a class is
    empty (undefined, reported as chance)."""
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ordered = sorted(pairs, key=lambda p: p[0])
    # Assign average ranks (1-based) to handle ties correctly.
    ranks = [0.0] * len(ordered)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # average of 1-based ranks i..j
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    sum_pos_ranks = sum(r for r, (_, y) in zip(ranks, ordered) if y == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _prf_at_threshold(
    pairs: list[tuple[float, int]], threshold: float
) -> tuple[float, float, float]:
    """Precision / recall / F1 for predicting label=1 when score >= threshold."""
    tp = sum(1 for s, y in pairs if s >= threshold and y == 1)
    fp = sum(1 for s, y in pairs if s >= threshold and y == 0)
    fn = sum(1 for s, y in pairs if s < threshold and y == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _best_f1(pairs: list[tuple[float, int]]) -> tuple[float, float, float, float]:
    """Sweep every distinct score as a threshold; return the F1-maximising
    (threshold, precision, recall, f1)."""
    thresholds = sorted({s for s, _ in pairs})
    best = (0.0, 0.0, 0.0, 0.0)
    for t in thresholds:
        p, r, f1 = _prf_at_threshold(pairs, t)
        if f1 > best[3]:
            best = (t, p, r, f1)
    return best


def _calibration(pairs: list[tuple[float, int]], n_bins: int = 5) -> list[dict]:
    """Bin scores into equal-width [0,1] bins; report predicted vs observed rate."""
    bins: list[dict] = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        members = [
            y for s, y in pairs
            if (s >= lo and (s < hi or (b == n_bins - 1 and s <= hi)))
        ]
        if not members:
            bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "observed_true_rate": None})
            continue
        bins.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": len(members),
                "observed_true_rate": round(sum(members) / len(members), 3),
            }
        )
    return bins


# ── Cohort generation (captures ground truth) ─────────────────────────────────

def _build_cohort_plan(kb_index: dict, safe_foods: list[str]) -> list[dict]:
    """Reproduce ``generate_synthetic_cohort``'s RNG walk (seed 42) so profiles are
    identical, but RETURN the ground-truth profile + diary for each patient instead
    of discarding them."""
    rng = random.Random(_COHORT_SEED)
    condition_options = [c for c, _ in SYNTHETIC_CONDITION_MIX]
    condition_weights = [w for _, w in SYNTHETIC_CONDITION_MIX]
    start_date = datetime.now(UTC) - timedelta(weeks=8)

    plan: list[dict] = []
    for i in range(_N_PATIENTS):
        primary = rng.choices(condition_options, weights=condition_weights, k=1)[0]
        conditions = [primary]
        if rng.random() < 0.30:
            others = [c for c in condition_options if c != primary]
            conditions.append(rng.choice(others))
        seed = rng.randint(0, 2**31 - 1)
        profile = generate_patient_profile(conditions, i, seed, kb_index, safe_foods)
        diary = generate_patient_diary(profile, kb_index, start_date)
        plan.append({"profile": profile, "diary": diary, "conditions": conditions})
    return plan


def _truth_components(profile: dict) -> set:
    """Ground-truth ComponentType set for a patient (profile trigger components
    mapped through the same COMPONENT_STR_TO_ENUM the seeder uses, e.g.
    sorbitol/fodmap -> FODMAP)."""
    truth = set()
    for comp_str in profile["trigger_components"]:
        ctype = COMPONENT_STR_TO_ENUM.get(comp_str)
        if ctype is not None:
            truth.add(ctype)
    return truth


# ── Per-patient evaluation ────────────────────────────────────────────────────

async def _eval_patient(db, user_id, truth: set, pop_prior) -> list[dict]:
    """Run the engine for one patient; emit one record per candidate component with
    both model scores. Frequentist score is derived from the engine's own 2x2 (the
    component analog of insights ``n_symptom_episodes / total_symptom_events``):
    ``a / (a + c)`` = fraction of symptom-outcome days that were exposed to the
    component. It shares the engine's exposure/outcome derivation exactly, so the
    only difference from the Bayesian score is prior-vs-raw-proportion."""
    results = await analyze_bayesian_triggers(
        db, user_id, lookback_days=90, population_prior=pop_prior
    )
    records: list[dict] = []
    scored: set = set()
    for r in results:
        scored.add(r.component_type)
        symptom_days = r.a + r.c
        freq = (r.a / symptom_days) if symptom_days > 0 else 0.0
        records.append(
            {
                "patient": str(user_id),
                "component": r.component_type.value,
                "component_type": r.component_type,
                "label": 1 if r.component_type in truth else 0,
                "bayes": r.trigger_probability,
                "freq": freq,
                "a": r.a, "b": r.b, "c": r.c, "d": r.d,
                "is_cold_start": r.is_cold_start,
            }
        )
    # Truth components the engine never scored (never eaten above threshold) = misses;
    # both models get score 0 so recall is penalised honestly.
    for ctype in truth - scored:
        records.append(
            {
                "patient": str(user_id),
                "component": ctype.value,
                "component_type": ctype,
                "label": 1,
                "bayes": 0.0,
                "freq": 0.0,
                "a": 0, "b": 0, "c": 0, "d": 0,
                "is_cold_start": None,
                "unscored_truth": True,
            }
        )
    return records


async def _run_once(db, user_ids, truths, pop_prior) -> list[dict]:
    all_records: list[dict] = []
    for user_id, truth in zip(user_ids, truths):
        all_records.extend(await _eval_patient(db, user_id, truth, pop_prior))
    return all_records


def _topk_hit_rate(records: list[dict], score_key: str, k: int) -> float:
    """Fraction of patients (with >=1 true trigger) whose top-k ranked components
    by ``score_key`` include at least one true trigger. This is the app's actual
    'suspect-foods leaderboard' success criterion — more clinically meaningful than
    pooled AUC. Ties broken by putting true triggers LAST (conservative)."""
    by_patient: dict[str, list[dict]] = {}
    for r in records:
        by_patient.setdefault(r["patient"], []).append(r)
    hits, evaluable = 0, 0
    for recs in by_patient.values():
        if not any(r["label"] == 1 for r in recs):
            continue
        evaluable += 1
        ranked = sorted(recs, key=lambda r: (r[score_key], -r["label"]), reverse=True)
        if any(r["label"] == 1 for r in ranked[:k]):
            hits += 1
    return hits / evaluable if evaluable else 0.0


def _head_to_head(records: list[dict]) -> dict:
    bayes_pairs = [(r["bayes"], r["label"]) for r in records]
    freq_pairs = [(r["freq"], r["label"]) for r in records]

    b_thr, b_p, b_r, b_f1 = _best_f1(bayes_pairs)
    f_thr, f_p, f_r, f_f1 = _best_f1(freq_pairs)

    # Fixed operating point: Bayesian P(exposed>unexposed) >= 0.5 ("more likely than not").
    fp_p, fp_r, fp_f1 = _prf_at_threshold(bayes_pairs, 0.5)

    return {
        "n_pairs": len(records),
        "n_positive": sum(r["label"] for r in records),
        "n_negative": sum(1 for r in records if r["label"] == 0),
        "bayesian": {
            "roc_auc": round(_roc_auc(bayes_pairs), 4),
            "best_f1_threshold": round(b_thr, 4),
            "precision": round(b_p, 4),
            "recall": round(b_r, 4),
            "f1": round(b_f1, 4),
            "fixed_thr_0.5": {
                "precision": round(fp_p, 4),
                "recall": round(fp_r, 4),
                "f1": round(fp_f1, 4),
            },
            "top1_hit_rate": round(_topk_hit_rate(records, "bayes", 1), 4),
            "top3_hit_rate": round(_topk_hit_rate(records, "bayes", 3), 4),
            "calibration": _calibration(bayes_pairs),
        },
        "frequentist": {
            "roc_auc": round(_roc_auc(freq_pairs), 4),
            "best_f1_threshold": round(f_thr, 4),
            "precision": round(f_p, 4),
            "recall": round(f_r, 4),
            "f1": round(f_f1, 4),
            "top1_hit_rate": round(_topk_hit_rate(records, "freq", 1), 4),
            "top3_hit_rate": round(_topk_hit_rate(records, "freq", 3), 4),
        },
    }


# ── Cold-start probe ──────────────────────────────────────────────────────────

async def _cold_start_probe(db, pop_prior) -> dict:
    """Create a throwaway IBS user with a declared condition but ZERO diary, then
    confirm the engine returns prior-only (is_cold_start) scores that are ELEVATED
    for condition-implicated components (FODMAP/LACTOSE/FRUCTOSE) vs a default
    component. Demonstrates the population-prior 'no empty shelf' behaviour."""
    probe_id = uuid.uuid4()
    db.add(
        User(
            id=probe_id,
            email=f"coldstart_{probe_id.hex[:8]}@foodai.internal",
            is_synthetic=True,
        )
    )
    await db.flush()
    db.add(UserCondition(user_id=probe_id, condition_type=ConditionType.IBS))
    await db.commit()

    results = await analyze_bayesian_triggers(
        db, probe_id, lookback_days=90, population_prior=pop_prior
    )
    by_comp = {r.component_type.value: r for r in results}

    # Prior-only scores for an implicated vs a non-implicated component.
    fodmap_rate = pop_prior.get(bt.ComponentType.FODMAP, bt.DEFAULT_POPULATION_RATE)
    other_rate = pop_prior.get(bt.ComponentType.OXALATES, bt.DEFAULT_POPULATION_RATE)
    implicated_prior = prior_score(fodmap_rate, implicated=True)
    default_prior = prior_score(other_rate, implicated=False)

    probe = {
        "probe_condition": "ibs",
        "all_results_cold_start": all(r.is_cold_start for r in results) if results else None,
        "n_candidate_components": len(results),
        "implicated_scores": {
            c: round(by_comp[c].score, 2)
            for c in ("fodmap", "lactose", "fructose")
            if c in by_comp
        },
        "prior_score_implicated_fodmap": round(implicated_prior, 2),
        "prior_score_default_component": round(default_prior, 2),
        "elevated_vs_default": implicated_prior > default_prior,
    }

    # Clean up the probe user.
    await db.execute(delete(UserCondition).where(UserCondition.user_id == probe_id))
    await db.execute(delete(User).where(User.id == probe_id))
    await db.commit()
    return probe


# ── Orchestration ─────────────────────────────────────────────────────────────

async def _reset_synthetic(db) -> None:
    """Delete any pre-existing synthetic users (cascade) so the population prior is
    built from exactly the cohort under test."""
    await db.execute(text("DELETE FROM users WHERE is_synthetic = true"))
    await db.commit()


async def _seed_cohort(db, plan: list[dict]) -> list[uuid.UUID]:
    from app.services.synthetic_data_generator import build_food_name_to_id_map

    food_map = await build_food_name_to_id_map(db)
    if not food_map:
        raise SystemExit(
            "food_database is empty — ingest the KB into the target DB first "
            "(see module docstring)."
        )
    user_ids: list[uuid.UUID] = []
    for entry in plan:
        uid = await insert_synthetic_patient(
            db, entry["profile"], entry["diary"], food_map, COMPONENT_STR_TO_ENUM
        )
        user_ids.append(uid)
    return user_ids


async def run(kb_path: str, report_path: str) -> dict:
    kb_index, safe_foods = load_kb_food_index(kb_path)
    print(f"KB: {len(kb_index)} trigger buckets, {len(safe_foods)} safe foods")

    plan = _build_cohort_plan(kb_index, safe_foods)
    truths = [_truth_components(e["profile"]) for e in plan]
    print(f"Cohort plan: {_N_PATIENTS} patients (seed {_COHORT_SEED})")

    async with async_session_factory() as db:
        await _reset_synthetic(db)
        user_ids = await _seed_cohort(db, plan)
        n_meals = sum(1 for e in plan for m in e["diary"]["meals"] if m["logged"])
        n_symp = sum(len(e["diary"]["symptoms"]) for e in plan)
        print(f"Seeded {len(user_ids)} patients | {n_meals} logged meals | {n_symp} symptoms")

        pop_prior = await build_population_prior_table(db)
        print(f"Population prior table: {len(pop_prior)} components")

        # ── Grid sweep: kappa x exposure threshold (Bayesian only, AUC/F1) ──────
        grid: list[dict] = []
        default_head_to_head = None
        for kappa in _KAPPA_GRID:
            for thr in _EXPOSURE_THRESHOLD_GRID:
                bt.PRIOR_STRENGTH_KAPPA = kappa
                bt.EXPOSURE_LEVEL_THRESHOLD = thr
                records = await _run_once(db, user_ids, truths, pop_prior)
                h2h = _head_to_head(records)
                cell = {
                    "kappa": kappa,
                    "exposure_threshold": thr,
                    "bayes_auc": h2h["bayesian"]["roc_auc"],
                    "bayes_f1": h2h["bayesian"]["f1"],
                    "freq_auc": h2h["frequentist"]["roc_auc"],
                    "freq_f1": h2h["frequentist"]["f1"],
                }
                grid.append(cell)
                print(
                    f"  kappa={kappa:<4} thr={thr:<4} | "
                    f"Bayes AUC={cell['bayes_auc']:.3f} F1={cell['bayes_f1']:.3f} | "
                    f"Freq AUC={cell['freq_auc']:.3f} F1={cell['freq_f1']:.3f}"
                )
                if kappa == _DEFAULT_KAPPA and thr == _DEFAULT_EXPOSURE:
                    default_head_to_head = h2h

        # Restore defaults before the cold-start probe.
        bt.PRIOR_STRENGTH_KAPPA = _DEFAULT_KAPPA
        bt.EXPOSURE_LEVEL_THRESHOLD = _DEFAULT_EXPOSURE

        cold_start = await _cold_start_probe(db, pop_prior)

    # Recommend the grid cell maximising Bayesian AUC, tie-broken by F1.
    best_cell = max(grid, key=lambda c: (c["bayes_auc"], c["bayes_f1"]))

    dflt_b = default_head_to_head["bayesian"]
    dflt_f = default_head_to_head["frequentist"]
    bayes_beats = (
        dflt_b["roc_auc"] >= dflt_f["roc_auc"] and dflt_b["f1"] >= dflt_f["f1"]
    )

    result = {
        "cohort": {
            "n_patients": _N_PATIENTS,
            "seed": _COHORT_SEED,
            "logged_meals": n_meals,
            "symptoms": n_symp,
            "population_prior_components": len(pop_prior),
        },
        "default_config": {"kappa": _DEFAULT_KAPPA, "exposure_threshold": _DEFAULT_EXPOSURE},
        "head_to_head_default": default_head_to_head,
        "bayesian_beats_frequentist": bayes_beats,
        "tuning_grid": grid,
        "recommended": {
            "kappa": best_cell["kappa"],
            "exposure_threshold": best_cell["exposure_threshold"],
            "bayes_auc": best_cell["bayes_auc"],
            "bayes_f1": best_cell["bayes_f1"],
        },
        "cold_start": cold_start,
        "caveats": [
            "IN-SAMPLE / CIRCULARITY: the population-prior table is aggregated from "
            "the SAME synthetic cohort we evaluate on, and the exposure derivation "
            "reads the SAME KB the generator used to plant triggers. Numbers are "
            "optimistic vs real patients; treat as an upper bound / sanity check, "
            "not a field estimate.",
            "THRESHOLD MISMATCH: the generator marks a food a component trigger at KB "
            "allergen score >= 20 (0-100), while the engine counts a day 'exposed' "
            "only at FoodComponentDetail.level >= 2.0 (0-4 'moderate+'). Low-level "
            "trigger foods planted by the generator are invisible to the engine, "
            "capping achievable recall for both models.",
            "SEEDING GAP: generate_synthetic_cohort inserts NO UserCondition rows "
            "(and 'food_allergy' has no ConditionType enum value), so on the seeded "
            "cohort the engine runs its NO-declared-condition path: no condition "
            "nudge, default 48h lag window, candidates = observed components only. "
            "This is the honest production behaviour for synthetic users; the "
            "condition-aware priors/lag are exercised only in the cold-start probe.",
            "Frequentist baseline = a/(a+c) on the engine's own 2x2 (component analog "
            "of the /insights suspect-foods proportion). It shares the engine's "
            "exposure/outcome derivation, isolating prior-vs-raw-proportion as the "
            "only difference; it is NOT the food-level endpoint verbatim.",
            "MULTI-COMPONENT CONFOUNDING (why de-confounding under-delivers here): the "
            "generator plants a symptom after a trigger FOOD but attributes it to one "
            "component, while that same food carries several components at level>=2 "
            "(e.g. a fructose-high fruit is also salicylate/histamine-high). The "
            "non-causal co-components are exposed on the exact same symptom days, so "
            "component-level de-confounding is ill-posed by construction — the "
            "Bayesian rate-contrast rewards broadly-present confounders (salicylates, "
            "additives, histamines) and is fooled MORE than the raw proportion. This "
            "is a property of the synthetic data, not proof the Bayesian is worse on "
            "real patients; but it means this cohort cannot demonstrate a "
            "de-confounding win.",
        ],
    }

    _write_report(result, report_path)
    _print_summary(result)
    return result


def _print_summary(result: dict) -> None:
    h = result["head_to_head_default"]
    print("\n" + "=" * 72)
    print("HEAD-TO-HEAD (default kappa=6, exposure_threshold=2.0)")
    print("=" * 72)
    print(f"{'metric':<22}{'Bayesian':>14}{'Frequentist':>16}")
    print("-" * 52)
    b, f = h["bayesian"], h["frequentist"]
    print(f"{'ROC-AUC':<22}{b['roc_auc']:>14.4f}{f['roc_auc']:>16.4f}")
    print(f"{'F1 (best-thr)':<22}{b['f1']:>14.4f}{f['f1']:>16.4f}")
    print(f"{'Precision (best-thr)':<22}{b['precision']:>14.4f}{f['precision']:>16.4f}")
    print(f"{'Recall (best-thr)':<22}{b['recall']:>14.4f}{f['recall']:>16.4f}")
    print(f"{'Top-1 hit rate':<22}{b['top1_hit_rate']:>14.4f}{f['top1_hit_rate']:>16.4f}")
    print(f"{'Top-3 hit rate':<22}{b['top3_hit_rate']:>14.4f}{f['top3_hit_rate']:>16.4f}")
    print(
        f"\nPairs: {h['n_pairs']} ({h['n_positive']} true triggers / "
        f"{h['n_negative']} non-triggers)"
    )
    print(f"Bayesian beats frequentist (AUC & F1): {result['bayesian_beats_frequentist']}")
    rec = result["recommended"]
    print(
        f"\nRECOMMENDED (max Bayesian AUC): kappa={rec['kappa']}, "
        f"exposure_threshold={rec['exposure_threshold']} "
        f"(AUC={rec['bayes_auc']:.3f}, F1={rec['bayes_f1']:.3f})"
    )
    cs = result["cold_start"]
    print(
        f"\nCold-start probe (IBS, no diary): all_cold_start={cs['all_results_cold_start']}, "
        f"implicated_scores={cs['implicated_scores']}, "
        f"elevated_vs_default={cs['elevated_vs_default']}"
    )
    print("=" * 72)


def _write_report(result: dict, report_path: str) -> None:
    h = result["head_to_head_default"]
    rec = result["recommended"]
    beats = result["bayesian_beats_frequentist"]
    report = {
        "report_version": 1,
        "agent": "backend-sprint",
        "wave": "Wave 2",
        "pillar": 1,
        "sprint": "Bayesian Sprint 2 — validate Bayesian vs frequentist on synthetic cohort",
        "status": "success",
        "timestamp": datetime.now(UTC).isoformat(),
        "summary": (
            f"Head-to-head on the 150-patient synthetic cohort: Bayesian ROC-AUC "
            f"{h['bayesian']['roc_auc']:.3f} / F1 {h['bayesian']['f1']:.3f} vs "
            f"frequentist AUC {h['frequentist']['roc_auc']:.3f} / F1 "
            f"{h['frequentist']['f1']:.3f}. Bayesian >= frequentist on AUC & F1: "
            f"{beats}. Recommended Sprint-3 defaults: kappa={rec['kappa']}, "
            f"exposure_threshold={rec['exposure_threshold']}. Report-only; no engine "
            f"constants edited, no deploy."
        ),
        "metrics": {
            "n_patients": result["cohort"]["n_patients"],
            "cohort_seed": result["cohort"]["seed"],
            "logged_meals": result["cohort"]["logged_meals"],
            "symptoms": result["cohort"]["symptoms"],
            "bayesian_roc_auc": h["bayesian"]["roc_auc"],
            "bayesian_f1": h["bayesian"]["f1"],
            "bayesian_precision": h["bayesian"]["precision"],
            "bayesian_recall": h["bayesian"]["recall"],
            "frequentist_roc_auc": h["frequentist"]["roc_auc"],
            "frequentist_f1": h["frequentist"]["f1"],
            "frequentist_precision": h["frequentist"]["precision"],
            "frequentist_recall": h["frequentist"]["recall"],
            "bayesian_beats_frequentist": beats,
            "recommended_kappa": rec["kappa"],
            "recommended_exposure_threshold": rec["exposure_threshold"],
            "head_to_head_default": h,
            "tuning_grid": result["tuning_grid"],
            "cold_start": result["cold_start"],
        },
        "blockers": [],
        "next_actions": [
            {
                "agent": "backend-sprint",
                "input": report_path,
                "condition": "status==success",
                "human_gate": False,
            }
        ],
        "artifacts": [
            "backend/scripts/validate_bayesian_vs_frequentist.py",
            report_path,
        ],
        "open_questions": [
            {
                "id": "BAYES-Q1",
                "question": (
                    "Adopt recommended kappa/exposure_threshold as engine defaults in "
                    "Sprint 3? (This PR does NOT edit the constants.)"
                ),
                "context": (
                    f"Grid max Bayesian AUC at kappa={rec['kappa']}, "
                    f"exposure_threshold={rec['exposure_threshold']}."
                ),
            },
            {
                "id": "BAYES-Q2",
                "question": (
                    "Should generate_synthetic_cohort insert UserCondition rows so the "
                    "engine's condition-aware priors/lag are exercised on synthetic "
                    "users? Needs a ConditionType for 'food_allergy' first."
                ),
                "context": "See caveats: seeding gap.",
            },
        ],
        "caveats": result["caveats"],
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nReport written: {report_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bayesian vs frequentist trigger validation.")
    p.add_argument("--kb-path", default=_DEFAULT_KB_PATH)
    p.add_argument("--report-path", default=_DEFAULT_REPORT_PATH)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.kb_path, args.report_path))
