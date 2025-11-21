# src/services/recommendation_service.py

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

class RecommendationService:
    def __init__(self):
        self.interaction_service = InteractionService()
        self.nutrient_service = NutrientService()

        # Try loading trained ML model
        try:
            self._load_model()
            print("✔ Model loaded successfully")
        except:
            print("⚠ Model load failed — retraining...")
            self._prepare_ml_data()
            self._save_model()
            print("✔ Model retrained and saved")

    # ---------------------------- ML DATA PREPARATION ----------------------------
    def _prepare_ml_data(self):
        """Prepare features and labels for ML training."""
        self.food_keys = list(self.nutrient_service.profile_map.keys())
        feature_list = []
        labels = []

        # Precompute all drug interactions
        all_drugs = self.interaction_service.df['drug_name'].unique()
        drug_interactions = {drug: set(self.interaction_service.get_drug_interactions(drug)) for drug in all_drugs}

        for fk in self.food_keys:
            profile = self.nutrient_service.profile_map[fk]

            # Extract numeric features
            numeric_values = [float(v) for v in profile.values() if isinstance(v, (int, float))]
            feature_list.append(numeric_values)

            # Label = 1 if any nutrient interacts with any drug
            unsafe = any(any(n in profile for n in nutrients) for nutrients in drug_interactions.values())
            labels.append(int(unsafe))

        # Pad numeric features to equal length
        max_len = max(len(f) for f in feature_list)
        self.feature_matrix = np.array([f + [0]*(max_len - len(f)) for f in feature_list])
        self.y = np.array(labels)

        # Scale features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.feature_matrix)

        # Train Random Forest
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_scaled, self.y)

        # Map food keys to row indices
        self.food_index_map = {fk: i for i, fk in enumerate(self.food_keys)}

    # ---------------------------- MODEL LOAD/SAVE ----------------------------
    def _load_model(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        with open(INDEX_MAP_PATH, "r") as f:
            self.food_index_map = json.load(f)

        # Reconstruct feature_matrix from nutrient_service
        feature_list = []
        for fk in self.food_index_map:
            profile = self.nutrient_service.profile_map[fk]
            numeric_values = [float(v) for v in profile.values() if isinstance(v, (int, float))]
            feature_list.append(numeric_values)
        max_len = max(len(f) for f in feature_list)
        self.feature_matrix = np.array([f + [0]*(max_len - len(f)) for f in feature_list])
        self.X_scaled = self.scaler.transform(self.feature_matrix)

    def _save_model(self):
        import os
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        with open(INDEX_MAP_PATH, "w") as f:
            json.dump(self.food_index_map, f)

    # ---------------------------- SAFE FOOD RECOMMENDATION ----------------------------
    def recommend_safe_foods(self, drug_list, gender=None, age=None):
        """Return foods safe across multiple drugs, optionally filtered by RDI."""
        if isinstance(drug_list, str):
            drug_list = [drug_list]

        # Aggregate all interacting nutrients
        bad_nutrients = set()
        for drug in drug_list:
            bad_nutrients.update(self.interaction_service.get_drug_interactions(drug))

        safe_items = []
        for fk, profile in self.nutrient_service.profile_map.items():
            if any(n in bad_nutrients for n in profile):
                continue

            # Optional RDI check
            if gender and age:
                rdi = self.nutrient_service.get_recommended_intake(gender, age)
                meets_rdi = all(profile.get(n, 0) >= min_val for n, (min_val, _) in rdi.items())
                if not meets_rdi:
                    continue

            safe_items.append({
                "food_key": fk,
                "food_name": self.nutrient_service.food_name_map[fk]
            })

        return safe_items

    # ---------------------------- ML PREDICTION ----------------------------
  
    def predict_food_safety(self, food_key):
        """ML predicts if a single food is safe (True) or unsafe (False)."""
        idx = self.food_index_map.get(food_key)
        if idx is None:
            return None
        row_features = self.feature_matrix[idx]
        scaled = self.scaler.transform([row_features])
        pred = self.model.predict(scaled)
        # Flip logic: 0 = safe, 1 = unsafe
        return bool(pred[0] == 0)  # True if safe

    def predict_proba(self, food_key):
        """Probability that a food is safe (0=safe, 1=unsafe)."""
        idx = self.food_index_map.get(food_key)
        if idx is None:
            return None
        row_features = self.feature_matrix[idx]
        scaled = self.scaler.transform([row_features])
        proba = self.model.predict_proba(scaled)[0]

        classes = list(self.model.classes_)
        if 0 in classes:
            safe_index = classes.index(0)  # index of "safe" class
            return float(proba[safe_index])
        else:
        # Only unsafe class seen, probability of safe = 0
            return 0.0

    

    # ---------------------------- DANGEROUS FOODS ----------------------------
    def list_dangerous_foods(self, drug_list):
        """Return foods potentially unsafe for a list of drugs."""
        if isinstance(drug_list, str):
            drug_list = [drug_list]

        bad_nutrients = set()
        for drug in drug_list:
            bad_nutrients.update(self.interaction_service.get_drug_interactions(drug))

        dangerous_items = []
        for fk, profile in self.nutrient_service.profile_map.items():
            if any(n in bad_nutrients for n in profile):
                dangerous_items.append({
                    "food_key": fk,
                    "food_name": self.nutrient_service.food_name_map[fk],
                    "ML_safe": self.predict_food_safety(fk),
                    "Probability": self.predict_proba(fk)
                })

        dangerous_items.sort(key=lambda x: x["Probability"] if x["Probability"] is not None else 0)
        return dangerous_items

# ------------------- MANUAL TEST -------------------
if __name__ == "__main__":
    drugs = ["Warfarin", "Aspirin", "Clopidogrel", "Ibuprofen", "Heparin",
             "Metformin", "Atorvastatin", "Omeprazole", "Lisinopril", "Simvastatin"]

    rec = RecommendationService()

    safe_foods = rec.recommend_safe_foods(drugs)
    print(f"✔ Top safe foods for {drugs}:")
    for i, food in enumerate(safe_foods[:10], 1):
        fk = food["food_key"]
        print(f"{i}. {food['food_name']} (ML safe: {rec.predict_food_safety(fk)}, Probability: {rec.predict_proba(fk):.2f})")

    dangerous_foods = rec.list_dangerous_foods(drugs)
    print(f"\n❌ Top potentially dangerous foods for {drugs}:")
    for i, food in enumerate(dangerous_foods[:10], 1):
        print(f"{i}. {food['food_name']} (ML safe: {food['ML_safe']}, Probability: {food['Probability']:.2f})")
