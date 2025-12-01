# src/services/interaction_service.py

import json
from pathlib import Path

# Your actual file based on your folder structure
DRUG_FILE = Path("dataset/drug_food_interaction/drugbank_interaction.json")


class InteractionService:
    """
    Loads DrugBank food interaction data from JSON
    and provides simple keyword extraction.
    """

    def __init__(self):
        if not DRUG_FILE.exists():
            raise FileNotFoundError(f"DrugBank JSON file not found at {DRUG_FILE}")

        # Load JSON: a list of objects like:
        # { "name": "Lepirudin", "food_interactions": ["Avoid garlic", ...] }
        with open(DRUG_FILE, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Normalize drug names → interactions
        self.drug_map = {}
        for drug in self.data:
            name = drug.get("name", "").lower().strip()
            items = drug.get("food_interactions", [])
            # Convert all interaction texts to lowercase
            self.drug_map[name] = [i.lower() for i in items]

    def get_drug_interactions(self, drug_name: str):
        """
        Returns a list of food interaction texts for a given drug.
        Example: ["avoid garlic", "avoid ginger"]

        ALWAYS in lowercase.
        """
        key = drug_name.lower().strip()
        return self.drug_map.get(key, [])

    def get_all_drug_names(self):
        """
        Returns list of all drug names in lowercase.
        """
        return list(self.drug_map.keys())
