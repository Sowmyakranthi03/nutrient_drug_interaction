from src.etl.extract import load_fsanz, load_drugbank
from src.etl.transform_food import transform_fsanz
from src.etl.transform_interactions import transform_drug_interactions
from src.etl.load import save_cleaned_outputs
from src.utils.logger import logger

if __name__ == "__main__":
    logger.info("Starting pipeline...")

    fsanz = load_fsanz()
    drug = load_drugbank()

    food_table = transform_fsanz(fsanz)
    drug_table = transform_drug_interactions(drug)

    save_cleaned_outputs(food_table, drug_table)

    logger.info("Pipeline completed")
