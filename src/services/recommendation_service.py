# src/services/recommendation_service.py

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.services.interaction_service import InteractionService
from src.services.nutrient_service import NutrientService

MODEL_PATH = "models/food_safety_model.pkl"
SCALER_PATH = "models/food_scaler.pkl"
INDEX_MAP_PATH = "models/food_index_map.json"


class RecommendationService:
    def __init__(self):
        # Initialize services
        self.interaction_service = InteractionService()
        self.nutrient_service = NutrientService()

        # Build ML feature matrix from numeric nutrient values
        self._prepare_features()

        # Load or train ML model
        try:
            self._load_model()
            print("✔ Model loaded successfully")
        except Exception as e:
            print("⚠ Model load failed, retraining...", e)
            self._train_model()
            self._save_model()
            print("✔ Model retrained and saved")

    # ------------------------- Feature extraction -------------------------
    def _prepare_features(self):
        # Determine all numeric nutrient keys across all foods
        all_keys = set()
        for profile in self.nutrient_service.profile_map.values():
            for k, v in profile.items():
             if isinstance(v, (int, float)):
                    all_keys.add(k)
        self.numeric_keys = sorted(list(all_keys))  # fixed order for ML

    # Build feature matrix
        self.feature_matrix = []
        for fk in self.nutrient_service.profile_map:
            profile = self.nutrient_service.profile_map[fk]
            row = [profile.get(k, 0.0) for k in self.numeric_keys]  # missing = 0
            self.feature_matrix.append(row)
        self.feature_matrix = np.array(self.feature_matrix)

        self.food_keys = list(self.nutrient_service.profile_map.keys())
        self.food_index_map = {fk: i for i, fk in enumerate(self.food_keys)}
       

    # ------------------------- ML TRAINING -------------------------
    def _train_model(self):
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.feature_matrix)

        # Generate labels: 1 = interacts with any drug, 0 = safe
        y = []
        for fk in self.food_keys:
            profile = self.nutrient_service.profile_map[fk]
            interacts = False
            for drug in self.interaction_service.df['drug_name'].unique():
                interacting_nutrients = self.interaction_service.get_drug_interactions(drug)
                if any(n in profile for n in interacting_nutrients):
                    interacts = True
                    break
            y.append(int(interacts))
        y = np.array(y)

        # Train RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        self.X_scaled = X_scaled  # save scaled features

    # ------------------------- SAVE / LOAD MODEL -------------------------
    def _save_model(self):
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        with open(INDEX_MAP_PATH, "w") as f:
            json.dump(self.food_index_map, f)

    def _load_model(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        with open(INDEX_MAP_PATH, "r") as f:
            self.food_index_map = json.load(f)
        # Scale features
        self.X_scaled = self.scaler.transform(self.feature_matrix)

    # ------------------------- RULE-BASED SAFE FOOD -------------------------
    def recommend_safe_foods(self, drug_names, gender=None, age=None):
        """
        Return foods safe with a given drug or list of drugs.
        Filters foods that contain any nutrient interacting with any drug.
        """
        if isinstance(drug_names, str):
            drug_names = [drug_names]

        # Union of all interacting nutrients
        bad_nutrients = set()
        for drug in drug_names:
            bad_nutrients.update(self.interaction_service.get_drug_interactions(drug))

        safe_items = []
        for fk, profile in self.nutrient_service.profile_map.items():
            if any(n in bad_nutrients for n in profile):
                continue

            # Optionally filter by gender/age RDI
            if gender and age:
                rdi = self.nutrient_service.get_recommended_intake(gender, age)
                meets = all(profile.get(n, 0) >= min_val for n, (min_val, _) in rdi.items())
                if not meets:
                    continue

            safe_items.append({
                "food_key": fk,
                "food_name": self.nutrient_service.food_name_map[fk]
            })

        return safe_items

    # ------------------------- ML PREDICTION -------------------------
    def predict_food_safety(self, food_key):
        """Return True if ML predicts food is safe, False if interacts, None if unknown."""
        idx = self.food_index_map.get(food_key)
        if idx is None:
            return None
        x = self.X_scaled[idx].reshape(1, -1)
        pred = self.model.predict(x)
        return bool(pred[0] == 0)

    def predict_food_proba(self, food_key):
        """Return probability that food is safe (0-1)."""
        idx = self.food_index_map.get(food_key)
        if idx is None:
            return None
        x = self.X_scaled[idx].reshape(1, -1)
        proba = self.model.predict_proba(x)[0][0]
        return float(proba)


# ------------------------- MANUAL TEST -------------------------
if __name__ == "__main__":
    rec = RecommendationService()

    drug_list = ["Warfarin", "Aspirin"]  # multiple drugs example
    print(f"Top safe foods for {drug_list}:")
    safe = rec.recommend_safe_foods(drug_list)
    print(safe[:10])

    first_food = list(rec.food_keys)[0]
    print("ML prediction for first food:", rec.predict_food_safety(first_food))
    print("ML probability for first food:", rec.predict_food_proba(first_food))
