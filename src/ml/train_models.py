# ------------------------------------------------------------
# TRAIN MACHINE LEARNING MODEL FOR FOOD SAFETY PREDICTION
# ------------------------------------------------------------

import json
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.services.interaction_service import InteractionService
from src.services.nutrient_service import NutrientService


OUTPUT_DIR = "models"
MODEL_PATH = f"{OUTPUT_DIR}/food_safety_model.pkl"
SCALER_PATH = f"{OUTPUT_DIR}/food_scaler.pkl"
INDEX_MAP_PATH = f"{OUTPUT_DIR}/food_index_map.json"


def train_model():
    print("🔍 Loading services...")
    ns = NutrientService()
    isvc = InteractionService()

    # Create model output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📦 Building dataset for ML...")

    all_food_keys = list(ns.profile_map.keys())
    numeric_features = []
    labels = []

    # Precompute drug interactions for speed
    drug_interactions = {
        drug: set(isvc.get_drug_interactions(drug))
        for drug in isvc.df["drug_name"].unique()
    }

    # Build feature matrix and labels
    for fk in all_food_keys:
        profile = ns.profile_map[fk]

        # Extract numeric values
        features = [v for v in profile.values() if isinstance(v, (int, float))]
        numeric_features.append(features)

        # Determine if food is UNSAFE (label=1)
        unsafe = False
        for nutrients in drug_interactions.values():
            if any(n in profile for n in nutrients):
                unsafe = True
                break

        labels.append(int(unsafe))

    # Normalize matrix size (pad)
    max_len = max(len(f) for f in numeric_features)
    X = np.array([f + [0] * (max_len - len(f)) for f in numeric_features])
    y = np.array(labels)

    print(f"📏 Feature matrix: {X.shape}, Labels: {y.shape}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🌲 Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_scaled, y)

    print("💾 Saving model files...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # Save food index map
    food_index_map = {fk: i for i, fk in enumerate(all_food_keys)}
    with open(INDEX_MAP_PATH, "w") as f:
        json.dump(food_index_map, f)

    print("🎉 Training completed successfully!")
    print("📁 Model saved in /models directory.")


if __name__ == "__main__":
    train_model()
