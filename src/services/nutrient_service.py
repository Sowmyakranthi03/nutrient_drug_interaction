# src/services/nutrient_service.py

import pandas as pd
import json

# Path to cleaned foods CSV from ETL
FOODS_FILE = "dataset/cleaned/foods_cleaned.csv"

class NutrientService:
    def __init__(self):
        # Load cleaned foods data
        self.df = pd.read_csv(FOODS_FILE)
        # Convert nutrient_profile column from JSON string to dict
        self.df['nutrient_profile'] = self.df['nutrient_profile'].apply(json.loads)

        # Map public_food_key -> nutrient_profile
        self.profile_map = {row['public_food_key']: row['nutrient_profile'] for _, row in self.df.iterrows()}
        # Map public_food_key -> food_name
        self.food_name_map = {row['public_food_key']: row['food_name'] for _, row in self.df.iterrows()}

        # Recommended Daily Intake (simplified example)
        # Nutrient -> (min_value, max_value) per gender/age group
        self.rdi = {
            'female_19-30': {'iron_g': (18, 20), 'protein_g': (46, 60)},
            'male_19-30': {'iron_g': (8, 10), 'protein_g': (56, 70)},
            # Add more gender/age groups as needed
        }

    def get_nutrient_profile(self, food_key: str):
        """Return nutrient profile dict for a given food key."""
        return self.profile_map.get(food_key, {})

    def get_foods_with_nutrient(self, nutrient: str, min_value: float = 0.0):
        """Return list of foods containing at least min_value of the nutrient."""
        result = []
        for key, profile in self.profile_map.items():
            if nutrient in profile and profile[nutrient] >= min_value:
                result.append((key, self.food_name_map.get(key), profile[nutrient]))
        return result

    def get_recommended_intake(self, gender: str, age: int):
        """Return recommended daily intake for a given gender and age."""
        # Simple age grouping example
        if 19 <= age <= 30:
            key = f"{gender}_19-30"
        else:
            key = f"{gender}_other"
        return self.rdi.get(key, {})

    def foods_meeting_rdi(self, gender: str, age: int):
        """Return foods that meet all recommended nutrient intakes for gender/age."""
        recommended = self.get_recommended_intake(gender, age)
        filtered_foods = []
        for fk, profile in self.profile_map.items():
            meets_rdi = all(
                profile.get(nutrient, 0) >= min_val
                for nutrient, (min_val, _) in recommended.items()
            )
            if meets_rdi:
                filtered_foods.append((fk, self.food_name_map[fk]))
        return filtered_foods

# Example usage
if __name__ == "__main__":
    ns = NutrientService()
    print("Loaded nutrient profiles:", len(ns.profile_map))
    print("Example nutrient profile for first food:")
    first_key = list(ns.profile_map.keys())[0]
    print(first_key, ns.get_nutrient_profile(first_key))

    # Foods meeting female 19-30 RDI
    foods_rdi = ns.foods_meeting_rdi("female", 25)
    print("Foods meeting female 19-30 RDI:", foods_rdi[:10])
