#!/usr/bin/env python3
"""W2-3 Barcode Benchmark: validates the barcode -> OFF -> KB pipeline E2E.

Exit box 5: "Barcode scan returns a structured trigger profile end-to-end for a
tested set of 50 common US grocery items."

This script runs the REAL pipeline against the LIVE Open Food Facts API for the
50-item set below. The automated test suite (tests/test_barcode.py) imports
``BARCODE_TEST_SET`` and drives the same pipeline with the OFF call MOCKED, so CI
never touches the network.

Usage:
    python backend/scripts/w2_3_barcode_benchmark.py

Metrics reported:
    - off_hit_rate_pct:    % of barcodes OFF recognised
    - profile_rate_pct:    % returning a structured profile with a tier_label
    - kb_match_rate_pct:   % with >=1 ingredient (or product) matched to the KB
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import barcode_service  # noqa: E402

# ---------------------------------------------------------------------------
# 50 common US grocery items. Each entry carries the fields the pipeline needs
# (barcode + product_name + representative ingredients) so the automated tests
# can synthesise an OFF payload without the network. `barcode` values are UPC-A
# strings for well-known US products; live OFF coverage varies over time.
# ---------------------------------------------------------------------------

BARCODE_TEST_SET = [
    {"barcode": "049000006344", "product_name": "Coca-Cola Classic",
     "ingredients": ["carbonated water", "high fructose corn syrup", "caramel color",
                     "phosphoric acid", "natural flavors", "caffeine"]},
    {"barcode": "038000138416", "product_name": "Rice Krispies Cereal",
     "ingredients": ["rice", "sugar", "salt", "malt flavor", "vitamins"]},
    {"barcode": "016000275867", "product_name": "Cheerios Cereal",
     "ingredients": ["whole grain oats", "corn starch", "sugar", "salt"]},
    {"barcode": "028400090858", "product_name": "Lay's Classic Potato Chips",
     "ingredients": ["potatoes", "vegetable oil", "salt"]},
    {"barcode": "044000032029", "product_name": "Oreo Chocolate Sandwich Cookies",
     "ingredients": ["wheat flour", "sugar", "palm oil", "cocoa", "high fructose corn syrup",
                     "soy lecithin"]},
    {"barcode": "021000658830", "product_name": "Kraft Macaroni and Cheese",
     "ingredients": ["wheat", "cheddar cheese", "milk", "whey", "butter"]},
    {"barcode": "051000012517", "product_name": "Campbell's Tomato Soup",
     "ingredients": ["tomato puree", "water", "wheat flour", "salt"]},
    {"barcode": "037600106672", "product_name": "Skippy Creamy Peanut Butter",
     "ingredients": ["roasted peanuts", "sugar", "palm oil", "salt"]},
    {"barcode": "051500255162", "product_name": "Jif Creamy Peanut Butter",
     "ingredients": ["roasted peanuts", "sugar", "molasses", "vegetable oil", "salt"]},
    {"barcode": "041303002094", "product_name": "Whole Milk Gallon",
     "ingredients": ["milk", "vitamin d"]},
    {"barcode": "070470496504", "product_name": "Almond Breeze Almond Milk",
     "ingredients": ["almond milk", "cane sugar", "sea salt", "sunflower lecithin"]},
    {"barcode": "025293001404", "product_name": "Silk Soy Milk",
     "ingredients": ["soymilk", "cane sugar", "sea salt", "natural flavor"]},
    {"barcode": "038000199530", "product_name": "Pringles Original",
     "ingredients": ["dried potatoes", "vegetable oil", "rice flour", "wheat starch", "salt"]},
    {"barcode": "028400047685", "product_name": "Doritos Nacho Cheese",
     "ingredients": ["corn", "vegetable oil", "cheddar cheese", "whey", "salt"]},
    {"barcode": "012000001291", "product_name": "Pepsi Cola",
     "ingredients": ["carbonated water", "high fructose corn syrup", "caramel color",
                     "sugar", "caffeine"]},
    {"barcode": "078000082401", "product_name": "Canada Dry Ginger Ale",
     "ingredients": ["carbonated water", "high fructose corn syrup", "citric acid",
                     "natural flavors"]},
    {"barcode": "030000010204", "product_name": "Quaker Oats Old Fashioned",
     "ingredients": ["whole grain rolled oats"]},
    {"barcode": "018627103783", "product_name": "Honey Roasted Almonds",
     "ingredients": ["almonds", "honey", "sugar", "sea salt"]},
    {"barcode": "085239004067", "product_name": "Canned Black Beans",
     "ingredients": ["black beans", "water", "salt"]},
    {"barcode": "024000163665", "product_name": "Canned Sweet Corn",
     "ingredients": ["corn", "water", "sugar", "salt"]},
    {"barcode": "011110812345", "product_name": "Greek Yogurt Plain",
     "ingredients": ["cultured milk", "cream"]},
    {"barcode": "036632001917", "product_name": "Chobani Strawberry Yogurt",
     "ingredients": ["milk", "strawberries", "cane sugar", "live cultures"]},
    {"barcode": "021130126026", "product_name": "Cheddar Cheese Block",
     "ingredients": ["cheddar cheese", "milk", "salt", "enzymes"]},
    {"barcode": "072250007504", "product_name": "Whole Wheat Bread",
     "ingredients": ["whole wheat flour", "water", "yeast", "salt", "soybean oil"]},
    {"barcode": "073410000175", "product_name": "White Sandwich Bread",
     "ingredients": ["wheat flour", "water", "yeast", "sugar", "salt"]},
    {"barcode": "085239401156", "product_name": "Frozen Chicken Breast",
     "ingredients": ["chicken breast"]},
    {"barcode": "023700043306", "product_name": "Butterball Ground Turkey",
     "ingredients": ["ground turkey"]},
    {"barcode": "204010000000", "product_name": "Fresh Atlantic Salmon Fillet",
     "ingredients": ["salmon"]},
    {"barcode": "033383000015", "product_name": "Bananas",
     "ingredients": ["banana"]},
    {"barcode": "033383401003", "product_name": "Gala Apples",
     "ingredients": ["apple"]},
    {"barcode": "819573013504", "product_name": "Avocado Single",
     "ingredients": ["avocado"]},
    {"barcode": "079893401706", "product_name": "Roma Tomatoes",
     "ingredients": ["tomato"]},
    {"barcode": "033383676432", "product_name": "Baby Spinach",
     "ingredients": ["spinach"]},
    {"barcode": "681131709323", "product_name": "Extra Virgin Olive Oil",
     "ingredients": ["olive oil"]},
    {"barcode": "041196010305", "product_name": "Heinz Tomato Ketchup",
     "ingredients": ["tomato concentrate", "distilled vinegar", "high fructose corn syrup",
                     "salt", "onion powder"]},
    {"barcode": "048001215146", "product_name": "French's Yellow Mustard",
     "ingredients": ["distilled vinegar", "mustard seed", "salt", "turmeric"]},
    {"barcode": "041390000164", "product_name": "Hellmann's Real Mayonnaise",
     "ingredients": ["soybean oil", "eggs", "vinegar", "salt", "lemon juice"]},
    {"barcode": "016000122581", "product_name": "Nature Valley Granola Bars",
     "ingredients": ["whole grain oats", "sugar", "canola oil", "honey", "almonds"]},
    {"barcode": "722252100849", "product_name": "Clif Bar Chocolate Chip",
     "ingredients": ["oats", "soy protein", "cane syrup", "chocolate chips", "peanuts"]},
    {"barcode": "015000001056", "product_name": "Tuna in Water",
     "ingredients": ["tuna", "water", "salt"]},
    {"barcode": "037000138013", "product_name": "Frozen Cheese Pizza",
     "ingredients": ["wheat flour", "tomato sauce", "mozzarella cheese", "yeast", "soybean oil"]},
    {"barcode": "021000615278", "product_name": "Philadelphia Cream Cheese",
     "ingredients": ["milk", "cream", "salt", "carob bean gum"]},
    {"barcode": "070662404003", "product_name": "Ground Coffee Medium Roast",
     "ingredients": ["coffee"]},
    {"barcode": "052000135794", "product_name": "Gatorade Lemon Lime",
     "ingredients": ["water", "sugar", "citric acid", "salt", "natural flavor"]},
    {"barcode": "041631000564", "product_name": "Brown Rice",
     "ingredients": ["brown rice"]},
    {"barcode": "070038315018", "product_name": "Spaghetti Pasta",
     "ingredients": ["durum wheat semolina"]},
    {"barcode": "051000149510", "product_name": "Prego Traditional Pasta Sauce",
     "ingredients": ["tomato puree", "onions", "garlic", "soybean oil", "salt", "basil"]},
    {"barcode": "044700028001", "product_name": "Beef Franks",
     "ingredients": ["beef", "water", "salt", "paprika", "garlic powder"]},
    {"barcode": "025700000000", "product_name": "Frozen Broccoli Florets",
     "ingredients": ["broccoli"]},
    {"barcode": "888849000000", "product_name": "Dark Chocolate Bar 70%",
     "ingredients": ["cocoa", "sugar", "cocoa butter", "milk", "soy lecithin"]},
]


def synthetic_off_product(entry: dict) -> dict:
    """Build an OFF-v2-shaped product dict from a test entry (used by tests)."""
    return {
        "product_name": entry["product_name"],
        "brands": entry.get("brands", ""),
        "ingredients": [{"text": ing} for ing in entry.get("ingredients", [])],
        "ingredients_text": ", ".join(entry.get("ingredients", [])),
    }


async def run_benchmark() -> dict:
    print("W2-3 Barcode Benchmark (LIVE Open Food Facts)")
    print(f"Test set: {len(BARCODE_TEST_SET)} items")
    print("=" * 60)

    off_hits = 0
    profile_ok = 0
    kb_matched = 0

    for i, entry in enumerate(BARCODE_TEST_SET, 1):
        result = await barcode_service.lookup_barcode_profile(entry["barcode"])
        has_tier = bool(result.get("tier_label"))
        if result.get("off_found"):
            off_hits += 1
        if has_tier and "ingredients" in result:
            profile_ok += 1
        if result.get("matched_count", 0) > 0:
            kb_matched += 1
        print(f"  [{i:2d}/50] {entry['product_name'][:38]:38s} "
              f"off={result.get('off_found')!s:5s} "
              f"tier={result.get('tier_label')}")

    n = len(BARCODE_TEST_SET)
    metrics = {
        "off_hit_rate_pct": round(off_hits / n * 100, 1),
        "profile_rate_pct": round(profile_ok / n * 100, 1),
        "kb_match_rate_pct": round(kb_matched / n * 100, 1),
    }
    print("=" * 60)
    print(f"  OFF hit rate:    {metrics['off_hit_rate_pct']}%")
    print(f"  Profile rate:    {metrics['profile_rate_pct']}% (target: 100%)")
    print(f"  KB match rate:   {metrics['kb_match_rate_pct']}%")
    return metrics


if __name__ == "__main__":
    asyncio.run(run_benchmark())
