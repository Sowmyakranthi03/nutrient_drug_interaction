# src/services/nutrient_service.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

# ---- NEW PATH LOGIC ----
# Base directory = project root (one level above "src")
BASE_DIR = Path(__file__).resolve().parents[2]
FSANZ_NUTRIENT_FILE = BASE_DIR /"dataset"/ "nutrient_databases" / "FSANZ" / "nutrient_file.xlsx"  

# Sheet names in your screenshot
SHEET_SOLIDS_100G = "All solids & liquids per 100g"
SHEET_LIQUIDS_100ML = "Liquids only per 100mL"


class NutrientService:
    """
    Loads FSANZ nutrient data in wide format.

    Exposes:
      - self.food_name_map: { food_key -> "Food Name" }
      - self.profile_map:   { food_key -> { nutrient_name -> value } }

    Nutrient values are per 100 g (or 100 mL for liquids sheet).
    """

    def __init__(self):
        if not FSANZ_NUTRIENT_FILE.exists():
            raise FileNotFoundError(f"FSANZ nutrient file not found at {FSANZ_NUTRIENT_FILE}")

        # Read both sheets if they exist, then concat
        dfs = []

        # Solids / general foods
        try:
            df_solids = pd.read_excel(
                FSANZ_NUTRIENT_FILE,
                sheet_name=SHEET_SOLIDS_100G,
            )
            dfs.append(df_solids)
        except ValueError:
            # sheet might not exist, ignore
            pass

        # Liquids per 100 mL (optional)
        try:
            df_liquids = pd.read_excel(
                FSANZ_NUTRIENT_FILE,
                sheet_name=SHEET_LIQUIDS_100ML,
            )
            dfs.append(df_liquids)
        except ValueError:
            pass

        if not dfs:
            raise ValueError(
                f"No usable sheets found in {FSANZ_NUTRIENT_FILE}. "
                f"Expected '{SHEET_SOLIDS_100G}' or '{SHEET_LIQUIDS_100ML}'."
            )

        df = pd.concat(dfs, ignore_index=True)

        # Normalise column names a bit
        df.columns = [str(c).strip() for c in df.columns]

        required = ["Public Food Key", "Food Name"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Expected column '{col}' in nutrient_file.xlsx")

        self.df = df

        # Build food name map
        self.food_name_map: Dict[str, str] = {}
        for _, row in df[["Public Food Key", "Food Name"]].dropna().iterrows():
            fk = str(row["Public Food Key"]).strip()
            self.food_name_map[fk] = str(row["Food Name"]).strip()

        # Nutrient columns = everything except identifiers
        id_cols = {"Public Food Key", "Food Name", "Classification"}
        nutrient_cols = [c for c in df.columns if c not in id_cols]

        # Build profile_map: food_key -> {nutrient_name -> value}
        self.profile_map: Dict[str, Dict[str, float]] = {}

        for _, row in df.iterrows():
            fk_raw = row.get("Public Food Key")
            if pd.isna(fk_raw):
                continue
            fk = str(fk_raw).strip()

            nutrient_profile: Dict[str, float] = {}
            for col in nutrient_cols:
                val = row.get(col)
                if pd.isna(val):
                    continue

                # Some values might come as strings with commas; coerce to float safely
                v = pd.to_numeric(val, errors="coerce")
                if pd.isna(v):
                    continue

                nutrient_profile[col] = float(v)

            # Only store if we have at least one nutrient
            if nutrient_profile:
                self.profile_map[fk] = nutrient_profile
            else:
                # still make sure food has an entry (empty dict)
                self.profile_map.setdefault(fk, {})

        print(
            f"✔ NutrientService loaded: {len(self.food_name_map)} foods, "
            f"{len(nutrient_cols)} nutrient columns"
        )

    # ------------------------------------------------------------------
    def get_recommended_intake(self, gender: str, age: int) -> Dict[str, tuple[float, float] | Any]:
        """
        Placeholder RDI function.

        Return mapping:
          nutrient_name -> (min_value_per_100g, max_value_per_100g or None)

        For now we return an empty dict so that RDI filtering
        does nothing. You can fill this with real values later.
        """
        return {}
