import pandas as pd
from pathlib import Path

DRUG_FILE = Path("dataset/cleaned/drugbank_interaction.csv")

class InteractionService:
    def __init__(self):
        self.df = pd.read_csv(DRUG_FILE)
        self.df.columns = [c.strip().lower() for c in self.df.columns]

    def get_drug_interactions(self, drug_name: str):
        drug_name = drug_name.strip().lower()
        interactions = self.df[self.df['drug_name'].str.lower() == drug_name]
        return interactions['interaction_text'].tolist()
