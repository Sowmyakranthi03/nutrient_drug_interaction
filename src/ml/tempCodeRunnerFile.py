# src/ml/train_models.py

import json
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.services.interaction_service import InteractionService
from src.services.nutrient_service import NutrientService

MODEL_PATH = "models/food_safety_model.pkl"
SCALER_PATH = "models/food_scaler.pkl"
INDEX_MAP_PATH = "models/food_index_map.json"


def prepare_training_data():
    interaction_service = InteractionService()
    nutrient_service = NutrientService()

    food_keys = list(nutrient_service.profile_map.keys())
    feature_list = []
    labels = []

    # Preload interactions for all drugs
    all_drugs = interaction_service.df["drug_name"].unique()
    drug_interactions = {
        drug: set(interaction_service.get_drug_interactions(drug))
        for drug in all_drugs
    }

    for fk in food_keys:
        profile = nutrient_service.profile_map[fk]

        # numeric features
        numeric_values = [
            float(v) for v in profile.values() if isinstance(v, (int, float))
        ]
        feature_list.append(numeric_values)

        # label logic: 1 = unsafe, 0 = safe
        unsafe = any(any(n in profile for n in bad) for bad in drug_interactions.values())
        labels.append(int(unsafe))

    # pad uneven lengths
    max_len = max(len(f) for f in feature_list)
    feature_matrix = np.array([f + [0] * (max_len - len(f)) for f in feature_list])
    y = np.array(labels)

    return food_keys, feature_matrix, y


def train_model():
    food_keys, X, y = prepare_training_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_scaled, y)

    food_index_map = {fk: i for i, fk in enumerate(food_keys)}

    # save files
    import os
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(INDEX_MAP_PATH, "w") as f:
        json.dump(food_index_map, f)

    print("\n✔ Training complete — model saved.")
    print(f"✔ Foods trained: {len(food_keys)}")
    print(f"✔ Unsafe ratio: {sum(y)}/{len(y)}")


if __name__ == "__main__":
    train_model()
