# src/services/recommendation_service.py

import pandas as pd
from src.services.interaction_service import InteractionService
from src.services.nutrient_service import NutrientService
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RecommendationService:
    def __init__(self):
        # Initialize dependent services
        self.interaction_service = InteractionService()
        self.nutrient_service = NutrientService()

        # Prepare dataset for ML-based recommendation (optional)
        self._prepare_ml_data()

    def _prepare_ml_data(self):
        """
        Prepare feature matrix X and labels y for ML recommendation.
        Here we generate a binary label: 1 if food contains nutrients that interact with drugs, 0 otherwise.
        """
        all_food_keys = list(self.nutrient_service.profile_map.keys())
        feature_list = []
        labels = []

        for fk in all_food_keys:
            profile = self.nutrient_service.get_nutrient_profile(fk)
            # Example: take numeric nutrient values as features
            features = [v for k, v in profile.items() if isinstance(v, (int, float))]
            feature_list.append(features)

            # Label = 1 if any nutrient interacts with any drug
            interacts = False
            for drug in self.interaction_service.df['drug_name'].unique():
                interacting_nutrients = self.interaction_service.get_drug_interactions(drug)
                if any(n in profile for n in interacting_nutrients):
                    interacts = True
                    break
            labels.append(int(interacts))

        # Convert to array (pad features to same length)
        max_len = max(len(f) for f in feature_list)
        X = np.array([f + [0]*(max_len - len(f)) for f in feature_list])
        y = np.array(labels)

        # Standardize features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        self.y = y

        # Train simple RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_scaled, self.y)

        # Map food keys to row index for later prediction
        self.food_index_map = {fk: i for i, fk in enumerate(all_food_keys)}

    def recommend_safe_foods(self, drug_name: str, gender: str = None, age: int = None):
        """
        Return a list of foods safe with a given drug, optionally filtered by recommended intake.
        """
        interacting_nutrients = set(self.interaction_service.get_drug_interactions(drug_name))
        safe_foods = []

        for fk, profile in self.nutrient_service.profile_map.items():
            # Skip if food has any interacting nutrient
            if any(n in interacting_nutrients for n in profile.keys()):
                continue

            # Optionally check gender/age RDI
            if gender and age:
                recommended = self.nutrient_service.get_recommended_intake(gender, age)
                meets_rdi = all(profile.get(n, 0) >= min_val for n, (min_val, _) in recommended.items())
                if not meets_rdi:
                    continue

            safe_foods.append((fk, self.nutrient_service.food_name_map[fk]))

        return safe_foods

    def predict_food_safety(self, food_key: str):
        """
        Use trained ML model to predict if a food is safe w.r.t. nutrient-drug interactions.
        Returns True if safe, False if potentially unsafe.
        """
        idx = self.food_index_map.get(food_key)
        if idx is None:
            return None
        x = self.X_scaled[idx].reshape(1, -1)
        pred = self.model.predict(x)
        return pred[0] == 0  # 0 = safe, 1 = interacts

# Example usage
if __name__ == "__main__":
    rec = RecommendationService()
    drug = "Warfarin"
    print(f"Top safe foods for {drug}:", rec.recommend_safe_foods(drug)[:10])

    # Predict with ML
    food_key = list(rec.nutrient_service.profile_map.keys())[0]
    print(f"ML predicts {rec.nutrient_service.food_name_map[food_key]} is safe:",
          rec.predict_food_safety(food_key))
