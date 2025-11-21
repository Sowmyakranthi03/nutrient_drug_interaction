# src/ml/train_models.py

import json
import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.services.interaction_service import InteractionService
from src.services.nutrient_service import NutrientService

MODEL_PATH = "models/food_safety_model.pkl"
SCALER_PATH = "models/food_scaler.pkl"
INDEX_MAP_PATH = "models/food_index_map.json"


def train_model():
    print("🔍 Loading services...")
    ns = NutrientService()
    isvc = InteractionService()

    os.makedirs("models", exist_ok=True)

    print("📦 Preparing feature matrix and labels...")

    food_keys = list(ns.profile_map.keys())
    feature_list = []
    labels = []

    # Precompute nutrient interactions per drug for efficiency
    all_drugs = isvc.df["drug_name"].unique()
    drug_interactions = {drug: set(isvc.get_drug_interactions(drug)) for drug in all_drugs}

    for fk in food_keys:
        profile = ns.profile_map[fk]

        # Numeric features only
        numeric_features = [float(v) for v in profile.values() if isinstance(v, (int, float))]
        feature_list.append(numeric_features)

        # Label = 1 if any nutrient interacts with any drug
        unsafe = any(any(n in profile for n in nutrients) for nutrients in drug_interactions.values())
        labels.append(int(unsafe))

    # Pad numeric features to same length
    max_len = max(len(f) for f in feature_list)
    X = np.array([f + [0] * (max_len - len(f)) for f in feature_list])
    y = np.array(labels)

    print(f"📏 Feature matrix shape: {X.shape}, Labels shape: {y.shape}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train RandomForest
    print("🌲 Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Save model, scaler, and index mapping
    print("💾 Saving model files...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    food_index_map = {fk: i for i, fk in enumerate(food_keys)}
    with open(INDEX_MAP_PATH, "w") as f:
        json.dump(food_index_map, f)

    print("🎉 Model trained and saved successfully!")


if __name__ == "__main__":
    train_model()
