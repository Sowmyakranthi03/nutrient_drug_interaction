import pandas as pd
import json
from src.config.config import FILES
from src.utils.logger import logger

def load_fsanz():
    logger.info("Loading FSANZ datasets...")

    data = {
        "food" : pd.read_excel(FILES["food_details"]),
        "nutrients" : pd.read_excel(FILES["nutrient_file"]),
        "measure" : pd.read_excel(FILES["measure_file"]),
        "recipe" : pd.read_excel(FILES["recipe_file"]),
        "retention" : pd.read_excel(FILES["retention_factors"]),
    }

    return data

def load_drugbank():
    logger.info("Loading DrugBank interactions...")

    with open(FILES["drugbank"], "r", encoding="utf8") as f:
        return json.load(f)
