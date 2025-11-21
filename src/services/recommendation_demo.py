# src/services/recommendation_demo.py

from src.services.recommendation_service import RecommendationService
import pandas as pd

def run_demo():
    print("\n===============================")
    print("   ML Recommendation Demo")
    print("===============================\n")

    rec = RecommendationService()

    # -----------------------------
    # 1. Select any 20 foods
    # -----------------------------
    foods = list(rec.nutrient_service.profile_map.items())[:20]

    results = []

    print("Testing ML predictions for first 20 foods...\n")

    for food_key, profile in foods:
        food_name = rec.nutrient_service.food_name_map.get(food_key, "Unknown")

        pred = rec.predict_food_safety(food_key)
        prob_safe = rec.predict_proba(food_key)

        results.append({
            "food_key": food_key,
            "food_name": food_name,
            "ml_safe": pred,
            "ml_safe_probability": round(prob_safe, 4)
        })

    df_results = pd.DataFrame(results)

    print(df_results.head(10).to_string(index=False))

    # -----------------------------
    # 2. Show top ML-safe foods
    # -----------------------------
    print("\n\nTop 10 safest foods (ML):")
    safe_sorted = df_results.sort_values("ml_safe_probability", ascending=False)
    print(safe_sorted.head(10).to_string(index=False))

    # -----------------------------
    # 3. Show top ML-risky foods
    # -----------------------------
    print("\nTop 10 risky foods (ML):")
    risky_sorted = df_results.sort_values("ml_safe_probability")
    print(risky_sorted.head(10).to_string(index=False))

    # -----------------------------
    # 4. Rule-based comparison
    # -----------------------------
    drug = "Warfarin"
    print(f"\n\nRule-based safe foods for {drug}:")
    safe_rule = rec.recommend_safe_foods(drug)
    
    print(safe_rule[:10])

    print("\nDemo completed successfully.\n")


if __name__ == "__main__":
    run_demo()
