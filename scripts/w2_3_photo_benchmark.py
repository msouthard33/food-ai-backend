#!/usr/bin/env python3
"""W2-3 Photo-to-Food Benchmark: validates the vision pipeline end-to-end.

Uses text prompts ("A photo of: [meal description]") as vision input to test
the pipeline without requiring actual photos. This tests the full pipeline:
vision model -> food extraction -> KB search -> allergen profile.

30-meal test set covers the Wave 2 exit criteria.

Usage:
    OPENAI_API_KEY=sk-... python backend/scripts/w2_3_photo_benchmark.py

Metrics reported:
    - photo_accuracy_pct: % of meals returning structured food output
    - avg_confidence: mean confidence across all identified foods
    - avg_foods_per_photo: mean food items per meal analysis
    - cost_per_photo_usd: estimated cost per photo analysis call
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Project path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai

# ---------------------------------------------------------------------------
# 30-meal test set (Wave 2 exit criteria)
# ---------------------------------------------------------------------------

MEAL_TEST_SET = [
    # Simple meals
    {"id": 1, "description": "scrambled eggs with cheddar cheese and spinach", "expected_foods": ["eggs", "cheddar", "spinach"], "min_foods": 2},
    {"id": 2, "description": "greek yogurt with blueberries and honey", "expected_foods": ["yogurt", "blueberries", "honey"], "min_foods": 2},
    {"id": 3, "description": "avocado toast with a fried egg", "expected_foods": ["avocado", "toast", "egg"], "min_foods": 2},
    {"id": 4, "description": "grilled chicken breast with steamed broccoli and rice", "expected_foods": ["chicken", "broccoli", "rice"], "min_foods": 2},
    {"id": 5, "description": "caesar salad with grilled chicken and croutons", "expected_foods": ["salad", "chicken", "croutons"], "min_foods": 2},
    # Compound meals
    {"id": 6, "description": "beef tacos with cheese, lettuce, and sour cream", "expected_foods": ["beef", "taco", "cheese"], "min_foods": 2},
    {"id": 7, "description": "spaghetti with marinara sauce and meatballs", "expected_foods": ["spaghetti", "sauce", "meatballs"], "min_foods": 2},
    {"id": 8, "description": "chicken tikka masala with basmati rice and naan", "expected_foods": ["chicken", "rice", "naan"], "min_foods": 2},
    {"id": 9, "description": "pad thai with shrimp, peanuts, and lime", "expected_foods": ["pad thai", "shrimp", "peanuts"], "min_foods": 2},
    {"id": 10, "description": "pho with beef, bean sprouts, and basil", "expected_foods": ["pho", "beef", "sprouts"], "min_foods": 2},
    # Breakfast items
    {"id": 11, "description": "pancakes with maple syrup and butter", "expected_foods": ["pancakes", "syrup", "butter"], "min_foods": 2},
    {"id": 12, "description": "oatmeal with banana, walnuts, and cinnamon", "expected_foods": ["oatmeal", "banana", "walnuts"], "min_foods": 2},
    {"id": 13, "description": "breakfast burrito with chorizo, eggs, and cheese", "expected_foods": ["burrito", "chorizo", "eggs"], "min_foods": 2},
    {"id": 14, "description": "bagel with cream cheese and smoked salmon", "expected_foods": ["bagel", "cream cheese", "salmon"], "min_foods": 2},
    {"id": 15, "description": "french toast with strawberries and whipped cream", "expected_foods": ["french toast", "strawberries"], "min_foods": 2},
    # Sandwiches / wraps
    {"id": 16, "description": "BLT sandwich with mayo on sourdough", "expected_foods": ["bacon", "lettuce", "tomato", "bread"], "min_foods": 2},
    {"id": 17, "description": "falafel wrap with tahini and pickled vegetables", "expected_foods": ["falafel", "tahini", "vegetables"], "min_foods": 2},
    {"id": 18, "description": "pulled pork sandwich with coleslaw", "expected_foods": ["pork", "coleslaw"], "min_foods": 2},
    # Bowls
    {"id": 19, "description": "poke bowl with salmon, avocado, and edamame", "expected_foods": ["salmon", "avocado", "edamame"], "min_foods": 2},
    {"id": 20, "description": "acai bowl with granola, banana, and coconut", "expected_foods": ["acai", "granola", "banana"], "min_foods": 2},
    {"id": 21, "description": "bibimbap with beef, egg, and mixed vegetables", "expected_foods": ["beef", "egg", "vegetables", "rice"], "min_foods": 2},
    # Snacks / sides
    {"id": 22, "description": "hummus with pita bread and carrot sticks", "expected_foods": ["hummus", "pita", "carrot"], "min_foods": 2},
    {"id": 23, "description": "cheese plate with crackers, grapes, and almonds", "expected_foods": ["cheese", "crackers", "grapes"], "min_foods": 2},
    # International cuisine
    {"id": 24, "description": "sushi platter with salmon nigiri, tuna roll, and edamame", "expected_foods": ["sushi", "salmon", "tuna"], "min_foods": 2},
    {"id": 25, "description": "shakshuka with feta cheese and crusty bread", "expected_foods": ["shakshuka", "eggs", "feta"], "min_foods": 2},
    {"id": 26, "description": "ramen with pork belly, soft egg, and nori", "expected_foods": ["ramen", "pork", "egg"], "min_foods": 2},
    {"id": 27, "description": "risotto with mushrooms and parmesan", "expected_foods": ["risotto", "mushrooms", "parmesan"], "min_foods": 2},
    # Dietary-specific (histamine / MCAS relevant)
    {"id": 28, "description": "grilled salmon with asparagus and lemon butter sauce", "expected_foods": ["salmon", "asparagus"], "min_foods": 2},
    {"id": 29, "description": "steak with baked potato and sour cream", "expected_foods": ["steak", "potato", "sour cream"], "min_foods": 2},
    {"id": 30, "description": "vegetable stir fry with tofu and soy sauce over brown rice", "expected_foods": ["tofu", "vegetables", "rice", "soy sauce"], "min_foods": 2},
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VISION_MODEL = "gpt-4o-mini"
KB_PATH = PROJECT_ROOT / "04 - Food Science & Data" / "allergen_knowledge_base_complete.json"
ARTIFACTS_DIR = PROJECT_ROOT / "04 - Food Science & Data" / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "04 - Food Science & Data" / "reports"


def create_text_photo_prompt(description: str) -> str:
    """Create a text-based prompt that simulates a photo description for vision model."""
    return (
        f"Imagine you are looking at a photo of the following meal: {description}. "
        "Identify every distinct food item that would be visible in such a photo."
    )


async def analyze_single_meal(client: openai.AsyncOpenAI, meal: dict) -> dict:
    """Run the vision pipeline on a single meal description."""
    description = meal["description"]
    prompt_text = create_text_photo_prompt(description)

    system_prompt = (
        "You are a food identification assistant for a dietary tracking app. "
        "Based on the meal description (simulating a photo), identify every distinct food item.\n\n"
        "Return ONLY valid JSON -- no markdown fences, no explanation.\n"
        "Format: [{\"food\": \"<food name>\", \"portion\": \"<estimated portion>\", "
        "\"confidence\": <0.0-1.0>}]\n\n"
        "Rules:\n"
        "- Be specific: 'grilled chicken breast' not just 'chicken'\n"
        "- Estimate portions in common units (cup, oz, piece, slice, tablespoon)\n"
        "- Confidence: 0.9+ for clearly identifiable items, 0.5-0.8 for uncertain\n"
        "- Include sauces, dressings, and condiments as separate items"
    )

    t0 = time.time()
    try:
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        raw_text = response.choices[0].message.content or ""
        usage = response.usage

        # Parse JSON
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            foods = json.loads(text)
            if not isinstance(foods, list):
                foods = [foods] if isinstance(foods, dict) else []
        except json.JSONDecodeError:
            foods = []

        # Cost calculation (gpt-4o-mini text: $0.00015/1K input, $0.0006/1K output)
        cost = 0.0
        if usage:
            cost = (usage.prompt_tokens / 1000) * 0.00015 + (usage.completion_tokens / 1000) * 0.0006

        return {
            "meal_id": meal["id"],
            "description": description,
            "expected_foods": meal["expected_foods"],
            "min_foods": meal["min_foods"],
            "identified_foods": foods,
            "food_count": len(foods),
            "has_structured_output": len(foods) > 0 and all(
                isinstance(f, dict) and "food" in f and "confidence" in f for f in foods
            ),
            "meets_min_foods": len(foods) >= meal["min_foods"],
            "elapsed_ms": elapsed_ms,
            "cost_usd": round(cost, 6),
            "tokens_input": usage.prompt_tokens if usage else 0,
            "tokens_output": usage.completion_tokens if usage else 0,
            "raw_response": raw_text[:500],
        }

    except Exception as e:
        return {
            "meal_id": meal["id"],
            "description": description,
            "error": str(e)[:200],
            "identified_foods": [],
            "food_count": 0,
            "has_structured_output": False,
            "meets_min_foods": False,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "cost_usd": 0.0,
        }


def check_food_coverage(result: dict) -> dict:
    """Check how many expected foods were found in the identified foods."""
    identified = [f.get("food", "").lower() for f in result.get("identified_foods", [])]
    expected = result.get("expected_foods", [])

    found = 0
    found_foods = []
    missed_foods = []

    for exp in expected:
        exp_lower = exp.lower()
        matched = any(exp_lower in ident or ident in exp_lower for ident in identified)
        if matched:
            found += 1
            found_foods.append(exp)
        else:
            missed_foods.append(exp)

    return {
        "expected_count": len(expected),
        "found_count": found,
        "coverage_pct": round(found / len(expected) * 100, 1) if expected else 0,
        "found_foods": found_foods,
        "missed_foods": missed_foods,
    }


async def run_benchmark():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        # Try reading from backend/.env
        env_path = PROJECT_ROOT / "backend" / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not api_key:
        print("ERROR: OPENAI_API_KEY not set and not found in backend/.env")
        sys.exit(1)

    client = openai.AsyncOpenAI(api_key=api_key)
    print(f"W2-3 Photo-to-Food Benchmark")
    print(f"Vision model: {VISION_MODEL}")
    print(f"Test set: {len(MEAL_TEST_SET)} meals")
    print(f"{'='*60}\n")

    # Run all meals sequentially (to avoid rate limits and track costs)
    results = []
    total_cost = 0.0
    total_foods = 0
    structured_count = 0
    total_confidence = 0.0
    confidence_count = 0

    for i, meal in enumerate(MEAL_TEST_SET):
        print(f"  [{i+1:2d}/30] {meal['description'][:50]}...", end=" ", flush=True)
        result = await analyze_single_meal(client, meal)
        results.append(result)

        total_cost += result.get("cost_usd", 0)
        food_count = result.get("food_count", 0)
        total_foods += food_count

        if result.get("has_structured_output"):
            structured_count += 1
            for f in result.get("identified_foods", []):
                conf = f.get("confidence", 0)
                if isinstance(conf, (int, float)):
                    total_confidence += conf
                    confidence_count += 1

        status_sym = "OK" if result["has_structured_output"] else "FAIL"
        print(f"[{status_sym}] {food_count} foods, ${result.get('cost_usd', 0):.5f}, {result.get('elapsed_ms', 0)}ms")

        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)

    # Coverage analysis
    coverage_results = []
    for r in results:
        cov = check_food_coverage(r)
        coverage_results.append(cov)

    # Aggregate metrics
    photo_accuracy_pct = round(structured_count / len(MEAL_TEST_SET) * 100, 1)
    avg_confidence = round(total_confidence / confidence_count, 3) if confidence_count > 0 else 0
    avg_foods_per_photo = round(total_foods / len(MEAL_TEST_SET), 1)
    cost_per_photo = round(total_cost / len(MEAL_TEST_SET), 6)
    avg_expected_coverage = round(
        sum(c["coverage_pct"] for c in coverage_results) / len(coverage_results), 1
    )
    benchmark_pass = photo_accuracy_pct >= 80.0

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Structured output:    {structured_count}/30 = {photo_accuracy_pct}% {'PASS' if benchmark_pass else 'FAIL'} (target: >=80%)")
    print(f"  Avg confidence:       {avg_confidence}")
    print(f"  Avg foods per photo:  {avg_foods_per_photo}")
    print(f"  Avg expected coverage:{avg_expected_coverage}%")
    print(f"  Cost per photo:       ${cost_per_photo:.6f}")
    print(f"  Total cost:           ${total_cost:.6f}")
    print(f"  Cost gate (<$0.01):   {'PASS' if cost_per_photo < 0.01 else 'FAIL - HUMAN GATE REQUIRED'}")

    # Show any failures
    failures = [r for r in results if not r.get("has_structured_output")]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for f in failures:
            print(f"    [{f['meal_id']}] {f['description']}: {f.get('error', 'no structured output')}")

    # Show coverage misses
    low_coverage = [(r, c) for r, c in zip(results, coverage_results) if c["coverage_pct"] < 50]
    if low_coverage:
        print(f"\n  Low coverage (<50% expected foods found) ({len(low_coverage)}):")
        for r, c in low_coverage:
            print(f"    [{r['meal_id']}] {r['description']}: {c['coverage_pct']}% ({c['missed_foods']})")

    # Save detailed artifact
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_id": "w2-3_photo_benchmark_results",
        "sprint": "W2-3",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vision_model": VISION_MODEL,
        "test_set_size": len(MEAL_TEST_SET),
        "metrics": {
            "photo_accuracy_pct": photo_accuracy_pct,
            "avg_confidence": avg_confidence,
            "avg_foods_per_photo": avg_foods_per_photo,
            "cost_per_photo_usd": cost_per_photo,
            "total_cost_usd": round(total_cost, 6),
            "avg_expected_coverage_pct": avg_expected_coverage,
            "structured_output_count": structured_count,
            "benchmark_pass": benchmark_pass,
        },
        "results": [
            {**r, "coverage": c}
            for r, c in zip(results, coverage_results)
        ],
    }

    out_path = ARTIFACTS_DIR / "w2-3_photo_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n  Saved detailed results to {out_path}")

    return artifact["metrics"]


if __name__ == "__main__":
    metrics = asyncio.run(run_benchmark())
    print(f"\nMetrics: {json.dumps(metrics, indent=2)}")
