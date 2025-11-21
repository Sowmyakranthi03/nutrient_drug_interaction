# tests/test_services.py
import sys
from pathlib import Path

# Add src folder to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from src.services.recommendation_service import RecommendationService


def main():
    rec_service = RecommendationService()

    # Example: safe foods for Warfarin
    safe_foods = rec_service.recommend_safe_foods("Warfarin")
    print(f"Top safe foods for Warfarin: {safe_foods[:10]}")

if __name__ == "__main__":
    main()
