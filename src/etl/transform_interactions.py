import json
import pandas as pd
from pathlib import Path


def flatten_drugbank_interactions(json_path: str) -> pd.DataFrame:
    """
    Transform drugbank_interaction.json into a clean dataframe:
    Columns:
        drug_name
        interaction_text
        interaction_type
        reference
    """
    print(f"Loading JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for entry in data:

        drug_name = entry.get("name")
        reference = entry.get("reference")
        interactions = entry.get("food_interactions", [])

        if isinstance(interactions, list):
            for text in interactions:
                rows.append({
                    "drug_name": drug_name,
                    "interaction_text": text,
                    "interaction_type": detect_interaction_type(text),
                    "reference": reference
                })

        # if food_interactions is missing or not a list → still create row
        elif isinstance(interactions, str):
            rows.append({
                "drug_name": drug_name,
                "interaction_text": interactions,
                "interaction_type": detect_interaction_type(interactions),
                "reference": reference
            })

    df = pd.DataFrame(rows)

    print(f"Flattened {len(df)} interaction rows.")

    return df


def detect_interaction_type(text: str) -> str:
    """
    Categorize interaction into types:
    avoid / caution / recommend / neutral
    """

    t = text.lower()

    if "avoid" in t:
        return "avoid"
    if "do not" in t:
        return "avoid"
    if "caution" in t:
        return "caution"
    if "recommend" in t or "administer" in t or "take" in t:
        return "recommended_intake"

    return "general_note"


def run_transformation():
    """
    Main ETL execution
    """

    input_path = r"d:\nutrient_drug_interaction\dataset\drug_food_interaction\drugbank_interaction.json"
    output_csv = r"d:\nutrient_drug_interaction\dataset\cleaned\drugbank_interaction_cleaned.csv"

    df = flatten_drugbank_interactions(input_path)

    # clean whitespace
    for col in ["drug_name", "interaction_text", "interaction_type", "reference"]:
        df[col] = df[col].astype(str).str.strip()

    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Saved cleaned interaction file → {output_csv}")


if __name__ == "__main__":
    run_transformation()
