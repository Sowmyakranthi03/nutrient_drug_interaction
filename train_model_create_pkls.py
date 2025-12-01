"""
FINAL PKL TRAINING SCRIPT — MATCHES YOUR PROJECT STRUCTURE EXACTLY
Runs once to generate:
    models/food_safety_model.pkl
    models/food_scaler.pkl
    models/food_index_map.json
    models/feature_order.json

Usage:
    python train_model_create_pkls.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------
# CORRECT PROJECT-ROOT BASED PATHS
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # project root

NUTRIENT_FILE = (
    BASE_DIR
    / "dataset"
    / "nutrient_databases"
    / "FSANZ"
    / "nutrient_file.xlsx"
)

DRUG_JSON_FILE = (
    BASE_DIR
    / "dataset"
    / "drug_food_interaction"
    / "drugbank_interaction.json"
)

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "food_safety_model.pkl"
SCALER_PATH = MODELS_DIR / "food_scaler.pkl"
INDEX_MAP_PATH = MODELS_DIR / "food_index_map.json"
FEATURE_ORDER_PATH = MODELS_DIR / "feature_order.json"


# -------------------------------------------------------------
# LOAD DRUGBANK JSON + EXTRACT FOOD KEYWORDS
# -------------------------------------------------------------
def load_drug_keywords():
    if not DRUG_JSON_FILE.exists():
        raise FileNotFoundError(f"DrugBank JSON not found at: {DRUG_JSON_FILE}")

    with open(DRUG_JSON_FILE, "r", encoding="utf-8") as f:
        drugs = json.load(f)

    # simple keyword discovery from food_interactions
    bad_keywords = set()

    for entry in drugs:
        interactions = entry.get("food_interactions", [])
        for line in interactions:
            text = line.lower()

            # naive keyword extraction based on actual used words
            for word in [
                "garlic", "ginger", "ginseng", "ginkgo", "chamomile",
                "echinacea", "bilberry", "danshen", "piracetam",
                "grapefruit", "alcohol", "wine", "beer",
                "milk", "dairy", "cheese", "yogurt",
                "coffee", "tea", "caffeine", "vitamin k", "vitamin c",
                "high-fat", "high fat",
            ]:
                if word in text:
                    bad_keywords.add(word)

    print("✔ Extracted bad keywords:", bad_keywords)
    return bad_keywords


# -------------------------------------------------------------
# LOAD FSANZ NUTRIENT FILE (MATCHES NUTRIENT SERVICE)
# -------------------------------------------------------------
def load_nutrient_df():
    if not NUTRIENT_FILE.exists():
        raise FileNotFoundError(f"Nutrient file not found: {NUTRIENT_FILE}")

    sheets = {}
    for sheet in ["All solids & liquids per 100g", "Liquids only per 100mL"]:
        try:
            df = pd.read_excel(NUTRIENT_FILE, sheet_name=sheet)
            sheets[sheet] = df
        except Exception:
            pass

    if not sheets:
        raise ValueError("No valid nutrient sheets found")

    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]

    if "Public Food Key" not in df.columns or "Food Name" not in df.columns:
        raise ValueError("Missing required FSANZ columns: Public Food Key, Food Name")

    print(f"✔ Nutrient file loaded: {df.shape[0]} foods, {df.shape[1]} columns")
    return df


# -------------------------------------------------------------
# BUILD FEATURE MATRIX + LABELS
# -------------------------------------------------------------
def build_dataset(df: pd.DataFrame, bad_keywords: set):
    food_keys = df["Public Food Key"].astype(str).tolist()
    food_names = df["Food Name"].astype(str).tolist()

    id_cols = {"Public Food Key", "Food Name", "Classification"}
    nutrient_cols = [c for c in df.columns if c not in id_cols]

    print(f"✔ Nutrient columns selected: {len(nutrient_cols)}")

    X = []
    y = []
    idx_map = {}

    for i, fk in enumerate(food_keys):
        name = food_names[i].lower()
        idx_map[fk] = i

        # label = unsafe if food name contains any bad keyword
        is_unsafe = any(term in name for term in bad_keywords)
        y.append(int(is_unsafe))

        # build nutrient vector
        row = []
        row_raw = df.iloc[i]
        for col in nutrient_cols:
            val = row_raw.get(col)
            try:
                row.append(float(val) if not pd.isna(val) else 0.0)
            except:
                row.append(0.0)
        X.append(row)

    print("✔ Label distribution:", dict(zip(*np.unique(y, return_counts=True))))
    return np.array(X, float), np.array(y, int), nutrient_cols, idx_map


# -------------------------------------------------------------
# TRAIN + SAVE PKL ARTIFACTS
# -------------------------------------------------------------
def train_and_save(X, y, nutrient_cols, idx_map):
    # if all safe/unsafe → model is meaningless
    if len(np.unique(y)) == 1:
        print("⚠ All labels identical. Skipping ML model.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=80,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(INDEX_MAP_PATH, "w") as f:
        json.dump(idx_map, f, indent=2)

    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(nutrient_cols, f, indent=2)

    print("🎉 MODEL TRAINING COMPLETE! Files saved in /models:")
    print(" -", MODEL_PATH)
    print(" -", SCALER_PATH)
    print(" -", INDEX_MAP_PATH)
    print(" -", FEATURE_ORDER_PATH)


# -------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------
if __name__ == "__main__":
    print("=========================================")
    print("  TRAINING FOOD SAFETY MODEL — PKL BUILDER")
    print("=========================================")

    bad_keywords = load_drug_keywords()
    df = load_nutrient_df()
    X, y, nutrient_cols, idx_map = build_dataset(df, bad_keywords)
    train_and_save(X, y, nutrient_cols, idx_map)
