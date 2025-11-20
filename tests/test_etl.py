# tests/test_etl.py
from pathlib import Path
import pandas as pd
import json

CLEANED_DIR = Path("dataset/cleaned")

def main():
    print("Testing ETL outputs...")

    # Check files exist
    expected_files = [
        "drugbank_interaction.csv",
        "foods_cleaned.csv",
        "foods_nutrients_long.csv",
        "foods_nutrients_profiles.json",
        "foods_nutrients_wide.csv",
        "nutrients_by_food.parquet",
        "recipes_expanded.csv"
    ]

    for f in expected_files:
        path = CLEANED_DIR / f
        assert path.exists(), f"{f} is missing!"
        print(f"{f} exists ✅")

    # Spot check one CSV
    foods_csv = CLEANED_DIR / "foods_cleaned.csv"
    df_foods = pd.read_csv(foods_csv)
    print(f"foods_cleaned.csv shape: {df_foods.shape}")
    print("Sample foods:", df_foods['food_name'].head(5).tolist())

    # Spot check nutrient profiles JSON
    profiles_json = CLEANED_DIR / "foods_nutrients_profiles.json"
    with open(profiles_json) as f:
        profiles = json.load(f)
    print(f"Loaded {len(profiles)} nutrient profiles")
    sample_keys = list(profiles.keys())[:5]
    for k in sample_keys:
        print(k, list(profiles[k].items())[:5])

if __name__ == "__main__":
    main()
