"""In-memory validation of synthetic trigger recovery.

Generates 10 synthetic patients, runs an in-memory correlation analysis
against their generated diaries, and computes precision/recall to verify
that the correlation algorithm can theoretically recover ground-truth
triggers from the synthetic data.

Does NOT insert any data to the database — safe to run at any time.

Usage:
    python -m scripts.validate_synthetic_triggers [--n-patients N]
"""

import argparse
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_KB_PATH = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "../../04 - Food Science & Data/allergen_knowledge_base_complete.json")
)

# Lag window covers all condition lag ranges (IBS: 2-36h, MCAS: 0.08-6h, allergy: 0.08-2h)
_CORRELATION_MIN_HOURS = 0.0
_CORRELATION_MAX_HOURS = 48.0


def _build_food_component_map(kb_index: dict) -> dict[str, set[str]]:
    """Invert kb_index to {food_name: set of component strings}."""
    food_to_components: dict[str, set[str]] = defaultdict(set)
    for comp_str, food_names in kb_index.items():
        for food_name in food_names:
            food_to_components[food_name].add(comp_str)
    return dict(food_to_components)


def _analyze_in_memory(
    diary: dict,
    food_to_components: dict[str, set[str]],
    min_hours: float = _CORRELATION_MIN_HOURS,
    max_hours: float = _CORRELATION_MAX_HOURS,
) -> dict[str, int]:
    """Simple in-memory correlation analysis.

    For each symptom event, looks at all meals within [min_hours, max_hours]
    before the symptom and counts how many times each trigger component
    appeared. Returns {component_str: correlation_count}.
    """
    correlations: dict[str, int] = defaultdict(int)

    for symptom in diary["symptoms"]:
        sym_ts = symptom["timestamp"]
        earliest = sym_ts - timedelta(hours=max_hours)
        latest = sym_ts - timedelta(hours=min_hours)

        for meal in diary["meals"]:
            meal_ts = meal["timestamp"]
            if earliest <= meal_ts <= latest:
                for food_name in meal["foods"]:
                    for comp in food_to_components.get(food_name, set()):
                        correlations[comp] += 1

    return dict(correlations)


def _top_n_components(correlations: dict[str, int], n: int = 3) -> list[str]:
    """Return the top-N components by correlation count."""
    sorted_items = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    return [comp for comp, _ in sorted_items[:n]]


def _precision_recall(predicted: list[str], ground_truth: list[str]) -> tuple[float, float]:
    """Compute precision and recall for predicted vs. ground-truth trigger sets."""
    pred_set = set(predicted)
    truth_set = set(ground_truth)
    tp = len(pred_set & truth_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(truth_set) if truth_set else 0.0
    return precision, recall


def run_validation(n_patients: int = 10, kb_path: str = _DEFAULT_KB_PATH) -> dict:
    """Generate n_patients in memory and evaluate trigger recovery.

    Returns:
        {
          "n_patients": int,
          "macro_precision": float,
          "macro_recall": float,
          "patient_results": [...]
        }
    """
    from app.services.synthetic_data_generator import (
        SYNTHETIC_CONDITION_MIX,
        generate_patient_diary,
        generate_patient_profile,
        load_kb_food_index,
    )

    print(f"Loading KB from {kb_path}")
    kb_index, safe_foods = load_kb_food_index(kb_path)
    food_to_components = _build_food_component_map(kb_index)
    print(f"KB: {len(kb_index)} trigger buckets, {len(safe_foods)} safe foods")

    rng = random.Random(99)
    condition_options = [c for c, _ in SYNTHETIC_CONDITION_MIX]
    condition_weights = [w for _, w in SYNTHETIC_CONDITION_MIX]
    start_date = datetime.now(timezone.utc) - timedelta(weeks=8)

    patient_results = []
    all_precisions = []
    all_recalls = []

    print(f"\nValidating {n_patients} synthetic patients in memory...")
    print("-" * 70)

    for i in range(n_patients):
        primary = rng.choices(condition_options, weights=condition_weights, k=1)[0]
        conditions = [primary]
        if rng.random() < 0.30:
            others = [c for c in condition_options if c != primary]
            conditions.append(rng.choice(others))

        seed = rng.randint(0, 2**31 - 1)
        profile = generate_patient_profile(conditions, i, seed, kb_index, safe_foods)
        diary = generate_patient_diary(profile, kb_index, start_date)

        # Ground truth: the profile's trigger_components
        ground_truth = profile["trigger_components"]

        # Run in-memory correlation
        correlations = _analyze_in_memory(diary, food_to_components)

        n_symptoms = len(diary["symptoms"])
        n_logged = sum(1 for m in diary["meals"] if m["logged"])

        if not correlations or n_symptoms == 0:
            # No correlations possible with no symptoms → skip from averages
            result = {
                "patient_index": i,
                "conditions": conditions,
                "ground_truth": ground_truth,
                "predicted_top3": [],
                "precision": None,
                "recall": None,
                "note": "no_symptoms",
            }
            patient_results.append(result)
            print(
                f"  Patient {i:3d} | {str(conditions):<40} | no symptoms generated — skipping"
            )
            continue

        predicted = _top_n_components(correlations, n=3)
        precision, recall = _precision_recall(predicted, ground_truth)
        all_precisions.append(precision)
        all_recalls.append(recall)

        result = {
            "patient_index": i,
            "conditions": conditions,
            "ground_truth": ground_truth,
            "predicted_top3": predicted,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "n_logged_meals": n_logged,
            "n_symptoms": n_symptoms,
            "top_correlations": dict(
                sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
        }
        patient_results.append(result)

        tp = len(set(predicted) & set(ground_truth))
        print(
            f"  Patient {i:3d} | {str(conditions):<38} | "
            f"symptoms={n_symptoms:3d} | "
            f"P={precision:.2f} R={recall:.2f} | "
            f"TP={tp}/{len(ground_truth)}"
        )

    macro_precision = sum(all_precisions) / max(len(all_precisions), 1)
    macro_recall = sum(all_recalls) / max(len(all_recalls), 1)
    f1 = (
        2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0
        else 0.0
    )

    print("-" * 70)
    print(f"\nValidation Summary ({len(all_precisions)} evaluable patients):")
    print(f"  Macro Precision:  {macro_precision:.3f}")
    print(f"  Macro Recall:     {macro_recall:.3f}")
    print(f"  F1 Score:         {f1:.3f}")
    print()
    print("Interpretation:")
    print("  Precision measures: of predicted triggers, how many are real?")
    print("  Recall measures:    of real triggers, how many were found?")
    print("  Low recall is expected — the algorithm uses a wide lag window")
    print("  and trigger foods co-appear with safe foods, creating noise.")
    print()

    return {
        "n_patients": n_patients,
        "n_evaluable": len(all_precisions),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "f1": round(f1, 3),
        "patient_results": patient_results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate synthetic trigger recovery in memory.")
    p.add_argument("--n-patients", type=int, default=10)
    p.add_argument("--kb-path", default=_DEFAULT_KB_PATH)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_validation(n_patients=args.n_patients, kb_path=args.kb_path)
    print(f"Final: precision={results['macro_precision']}, recall={results['macro_recall']}")
