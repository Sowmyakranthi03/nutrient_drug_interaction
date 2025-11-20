import os

BASE_DATASET_PATH = "dataset"

FSANZ_PATH = os.path.join(BASE_DATASET_PATH, "nutrient_databases", "FSANZ")
DRUGBANK_PATH = os.path.join(BASE_DATASET_PATH, "drug_food_interaction")
CLEANED_PATH = os.path.join(BASE_DATASET_PATH, "cleaned")

FILES = {
    "food_details": f"{FSANZ_PATH}/food_details.xlsx",
    "nutrient_file": f"{FSANZ_PATH}/nutrient_file.xlsx",
    "measure_file": f"{FSANZ_PATH}/measure_file.xlsx",
    "recipe_file": f"{FSANZ_PATH}/recipe_file.xlsx",
    "retention_factors": f"{FSANZ_PATH}/retention_factors.xlsx",
    "drugbank": f"{DRUGBANK_PATH}/drugbank_interaction.json",
}
