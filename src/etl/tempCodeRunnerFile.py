"""
transform_drug_interaction.py
Processes DrugBank-style food/drug interaction JSON into a clean tabular dataset.
"""

import os
import json
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

RAW_PATH = os.path.join(
    BASE_DIR,
    "dataset/drug_food_interaction/drugbank_interaction.json"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "cleaned")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------
# NLP helper functions
# -------------------------------------------------------

COMMON_FOOD_TERMS = [
    "food", "meal", "fat", "alcohol", "grapefruit", "juice",
    "coffee", "milk", "dairy", "tea", "fiber", "protein", "carbohydrate"
]


def extract_food_term(text: str) -> str:
    """
    Extracts the most likely food item from the interaction text.
    Uses regex + keyword search.
    """

    text_clean = text.lower()

    # 1. Check known food terms
    for term in COMMON_FOOD_TERMS:
        if term in text_clean:
            return term

    # 2. Extract nouns (simple heuristic)
    tokens = re.findall(r"\b[a-zA-Z]+\b", text_clean)

    # pick candidates that sound like foods
    for t in tokens:
        if t.endswith("fruit") or t.endswith("juice"):
            return t
        if t in ["cheese", "banana", "orange", "citrus"]:
            return t

    # fallback
    return "food"


def classify_interaction(text: str) -> str:
    """
    Classifies a drug-food interaction into a general category.
    """

    text_l = text.lower()

    if "increase" in text_l or "enhance" in text_l:
        return "increases absorption"

    if "decrease" in text_l or "reduce" in text_l:
        return "decreases absorption"

    if "avoid" in text_l:
        return "avoid"

    if "take with food" in text_l:
        return "take with food"

    if "take on empty stomach" in text_l:
        return "empty stomach"

    return "general"


# -------------------------------------------------------
# Main flatten function
# -------------------------------------------------------

def flatten_interactions(drug_record):
    """
    Converts each drug record into multiple rows (one per interaction).
    """

    drug_name = drug_record.get("name", "Unknown Drug")
    interactions = drug_record.get("food_interaction", [])

    rows = []

    for item in interactions:

        if not isinstance(item, str):
            continue

        food_item = extract_food_term(item)
        interaction_type = classify_interaction(item)

        rows.append({
            "drug_name": drug_name,
            "food_item": food_item,
            "interaction_text": item,
            "interaction_type": interaction_type
        })

    return rows


# -------------------------------------------------------
# Load JSON
# -------------------------------------------------------

def load_interaction_json(path):
    print(f"Loading JSON: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------

def run_transformation():
    print("\n=== PROCESSING Drug-Food Interaction Dataset ===")

    data = load_interaction_json(RAW_PATH)

    all_rows = []

    print("Flattening records ...")
    for record in data:
        all_rows.extend(flatten_interactions(record))

    df = pd.DataFrame(all_rows)

    print("Cleaning whitespace...")
    for col in ["drug_name", "food_item", "interaction_text", "interaction_type"]:
        df[col] = df[col].astype(str).str.strip()

    # Save CSV
    out_csv = os.path.join(OUTPUT_DIR, "drug_food_interactions_cleaned.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV → {out_csv}")

    # Try saving Parquet
    try:
        out_parquet = os.path.join(OUTPUT_DIR, "drug_food_interactions_cleaned.parquet")
        df.to_parquet(out_parquet, index=False)
        print(f"Saved Parquet → {out_parquet}")
    except Exception as e:
        print("\n⚠ WARNING: Could not save Parquet (install pyarrow).")
        print(e)

    print("\nDONE. Drug interaction dataset cleaned.\n")


# -------------------------------------------------------
# Entry point
# -------------------------------------------------------

if __name__ == "__main__":
    run_transformation()
