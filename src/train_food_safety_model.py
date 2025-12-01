from src.services.recommendation_service import RecommendationService

if __name__ == "__main__":
    rec = RecommendationService(force_retrain=True)
    print("✅ Training pipeline completed.")
