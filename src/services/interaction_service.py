import json
from typing import Dict, List
from pathlib import Path
from src.config.settings import DRUGBANK_JSON

class InteractionService:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else Path(DRUGBANK_JSON)
        self.drug_map: Dict[str, List[str]] = self._load()

    def _load(self) -> Dict[str, List[str]]:
        if not self.path.exists():
            raise FileNotFoundError(f"DrugBank JSON not found: {self.path}")

        data = json.loads(self.path.read_text(encoding="utf-8"))

        # expected: list[{name, food_interactions:[...]}]
        drug_map: Dict[str, List[str]] = {}
        for item in data:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            interactions = item.get("food_interactions") or []
            interactions = [str(x).strip() for x in interactions if str(x).strip()]
            drug_map[name.lower()] = interactions

        return drug_map

    def list_drugs(self) -> List[str]:
        return sorted([k for k in self.drug_map.keys()])

    def get_drug_interactions(self, drug_name: str) -> List[str]:
        if not drug_name:
            return []
        return self.drug_map.get(drug_name.strip().lower(), [])
