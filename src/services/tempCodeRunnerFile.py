from services.recommendation_service import RecommendationService
import json

def main():
    rec_service = RecommendationService()

    # Example 1: safe foods for a drug
    drug = "Warfarin"
    safe_foods = rec_service.recommend_safe_foods(drug)
    print(f"\nTop safe foods for {drug}:")
    for key, name in safe_foods:
        print(f"{key} - {name}")

    # Example 2: top foods high in a nutrient
    nutrient = "protein"
    top_foods = rec_service.recommend_foods_high_in(nutrient)
    print(f"\nTop {len(top_foods)} foods high in {nutrient}:")
    for key, name, value in top_foods:
        print(f"{key} - {name}: {value} g per 100g")

    # Example 3: check if a specific food interacts with a drug
    food_key = safe_foods[0][0] if safe_foods else None
    if food_key:
        interacting_nuts = rec_service.check_food_drug_interaction(food_key, drug)
        print(f"\nNutrients in {food_key} that interact with {drug}: {interacting_nuts}")

if __name__ == "__main__":
    main()
