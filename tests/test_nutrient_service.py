# tests/test_nutrient_service.py
from src.services.nutrient_service import NutrientService

def main():
    print("Testing NutrientService...")
    ns = NutrientService()

    # List 10 foods from nutrient profiles
    foods = list(ns.profile_map.keys())[:10]  # use profile_map instead of profiles
    print("First 10 foods from nutrient profiles:")
    for key in foods:
        print(key, ns.food_name_map[key])

    # Example: get nutrient profile for a food
    sample_food_key = foods[0]
    profile = ns.get_nutrient_profile(sample_food_key)
    print(f"\nNutrient profile for {sample_food_key} ({ns.food_name_map[sample_food_key]}):")
    for nutrient, value in list(profile.items())[:5]:  # show first 5 nutrients
        print(f"{nutrient}: {value}")

    # Example: get foods high in protein
    protein_rich_foods = ns.get_foods_with_nutrient("protein", min_value=10.0)
    print(f"\nFoods with protein >= 10g:")
    for key, name, value in protein_rich_foods[:5]:  # show top 5
        print(f"{key} - {name}: {value}g protein")

if __name__ == "__main__":
    main()
