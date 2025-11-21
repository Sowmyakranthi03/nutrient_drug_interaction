import pandas as pd
import json
import numpy as np

FOODS_FILE = "dataset/cleaned/foods_cleaned.csv"

class NutrientService:
    def __init__(self):
        self.df = pd.read_csv(FOODS_FILE)
        self.df['nutrient_profile'] = self.df['nutrient_profile'].apply(json.loads)

        # Map food keys -> nutrient profiles
        self.profile_map = {row['public_food_key']: row['nutrient_profile'] for _, row in self.df.iterrows()}
        self.food_name_map = {row['public_food_key']: row['food_name'] for _, row in self.df.iterrows()}

        # Recommended Daily Intake
        self.rdi = {
            'female_19-30': {'iron_g': (18, 20), 'protein_g': (46, 60)},
            'male_19-30': {'iron_g': (8, 10), 'protein_g': (56, 70)},
        }

        # -----------------------------
        # FEATURE MATRIX FOR ML
        # -----------------------------
        # Get all unique nutrient keys
        self.all_nutrients = sorted({k for profile in self.profile_map.values() for k in profile.keys()})
        # Build numeric feature matrix: rows = foods, columns = nutrients
        self.feature_matrix = np.array([
            [profile.get(n, 0) for n in self.all_nutrients]
            for profile in self.profile_map.values()
        ])

    def get_nutrient_profile(self, food_key: str):
        return self.profile_map.get(food_key, {})

    def get_foods_with_nutrient(self, nutrient: str, min_value: float = 0.0):
        result = []
        for key, profile in self.profile_map.items():
            if nutrient in profile and profile[nutrient] >= min_value:
                result.append((key, self.food_name_map.get(key), profile[nutrient]))
        return result

    def get_recommended_intake(self, gender: str, age: int):
        if 19 <= age <= 30:
            key = f"{gender}_19-30"
        else:
            key = f"{gender}_other"
        return self.rdi.get(key, {})

    def foods_meeting_rdi(self, gender: str, age: int):
        recommended = self.get_recommended_intake(gender, age)
        filtered_foods = []
        for fk, profile in self.profile_map.items():
            meets_rdi = all(profile.get(nutrient, 0) >= min_val for nutrient, (min_val, _) in recommended.items())
            if meets_rdi:
                filtered_foods.append((fk, self.food_name_map[fk]))
        return filtered_foods

# Example usage
if __name__ == "__main__":
    ns = NutrientService()
    print("Loaded nutrient profiles:", len(ns.profile_map))
    first_key = list(ns.profile_map.keys())[0]
    print("Example nutrient profile for first food:", first_key, ns.get_nutrient_profile(first_key))
    foods_rdi = ns.foods_meeting_rdi("female", 25)
    print("Foods meeting female 19-30 RDI:", foods_rdi[:10])
    print("Feature matrix shape:", ns.feature_matrix.shape)
