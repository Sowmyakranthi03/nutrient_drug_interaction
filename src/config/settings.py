from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # project root

DATASET_DIR = BASE_DIR / "dataset"

DRUGBANK_JSON = DATASET_DIR / "drug_food_interaction" / "drugbank_interaction.json"

FSANZ_DIR = DATASET_DIR / "nutrient_databases" / "FSANZ"
FSANZ_NUTRIENT_XLSX = FSANZ_DIR / "nutrient_file.xlsx"
FSANZ_FOOD_DETAILS_XLSX = FSANZ_DIR / "food_details.xlsx"

DB_DIR = BASE_DIR / "src" / "database"
SQLITE_PATH = DB_DIR / "feedback.db"
DB_DIR.mkdir(parents=True, exist_ok=True)
