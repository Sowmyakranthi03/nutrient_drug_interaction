# src/services/nutrient_service.py
from __future__ import annotations

import pandas as pd
from typing import Optional
from src.config.settings import FSANZ_NUTRIENT_XLSX, FSANZ_FOOD_DETAILS_XLSX


def _norm_col(c: str) -> str:
    c = str(c).replace("\n", " ").replace("\r", " ").strip()
    c = " ".join(c.split())
    return c


def _canon(s: str) -> str:
    """canonical form for matching columns"""
    return _norm_col(s).lower().replace(" ", "").replace("_", "").replace("-", "")


class NutrientService:
    """
    Loads FSANZ nutrient sheet + (optional) food_details sheet and builds a clean dataframe:
    Required columns in output:
      - Public Food Key
      - Food Name
      - Classification
    All other columns are numeric (NaN -> 0.0)
    """

    def __init__(self, nutrient_path: str | None = None, food_details_path: str | None = None):
        self.nutrient_path = nutrient_path or FSANZ_NUTRIENT_XLSX
        self.food_details_path = food_details_path or FSANZ_FOOD_DETAILS_XLSX
        self.df = self._load_and_build()

    # -----------------------------
    # sheet selection
    # -----------------------------
    def _pick_sheet(self, xls: pd.ExcelFile) -> str:
        # Prefer "All solids & liquids per 100g" (or any sheet with 100g)
        for s in xls.sheet_names:
            sl = s.lower()
            if "100g" in sl or "100 g" in sl:
                return s
        return xls.sheet_names[0]

    # -----------------------------
    # column standardization
    # -----------------------------
    def _standardize_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [_norm_col(c) for c in df.columns]

        canon_map = { _canon(c): c for c in df.columns }

        # Possible variants -> target
        want = {
            "publicfoodkey": "Public Food Key",
            "foodname": "Food Name",
            "foodname.": "Food Name",
            "foodname ": "Food Name",
            "foodname(per100g)": "Food Name",
            "classification": "Classification",
        }

        # also accept "Food name" exactly
        for k, target in list(want.items()):
            if k in canon_map:
                df = df.rename(columns={canon_map[k]: target})

        # Extra safety: try fuzzy-like fallbacks by contains
        if "Public Food Key" not in df.columns:
            for c in df.columns:
                if "public food key" in c.lower():
                    df = df.rename(columns={c: "Public Food Key"})
                    break

        if "Food Name" not in df.columns:
            for c in df.columns:
                if c.lower().strip() in ["food name", "foodname", "name", "food name."]:
                    df = df.rename(columns={c: "Food Name"})
                    break

        if "Classification" not in df.columns:
            for c in df.columns:
                if "classification" == _canon(c):
                    df = df.rename(columns={c: "Classification"})
                    break

        # Validate
        for must in ["Public Food Key", "Food Name", "Classification"]:
            if must not in df.columns:
                raise ValueError(f"Missing required column: {must} | found={list(df.columns)[:25]}...")

        return df

    # -----------------------------
    # load nutrient sheet
    # -----------------------------
    def _load_nutrient_sheet(self) -> pd.DataFrame:
        xls = pd.ExcelFile(self.nutrient_path)
        sheet = self._pick_sheet(xls)
        df = pd.read_excel(self.nutrient_path, sheet_name=sheet)

        df = self._standardize_required_columns(df)

        # Clean key + names
        df["Public Food Key"] = df["Public Food Key"].astype(str).str.strip()
        df["Food Name"] = df["Food Name"].astype(str).str.strip()
        df["Classification"] = df["Classification"].astype(str).str.strip()

        # Remove empty/nan names
        df = df[df["Food Name"].notna()]
        df = df[df["Food Name"].str.lower() != "nan"]
        df = df[df["Food Name"].str.len() > 0]

        # Remove empty keys too
        df = df[df["Public Food Key"].notna()]
        df = df[df["Public Food Key"].str.lower() != "nan"]
        df = df[df["Public Food Key"].str.len() > 0]

        return df.reset_index(drop=True)

    # -----------------------------
    # load food details (optional)
    # -----------------------------
    def _load_food_details(self) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_excel(self.food_details_path)
            df.columns = [_norm_col(c) for c in df.columns]

            # Standardize key and (optional) name
            if "Public Food Key" not in df.columns:
                for c in df.columns:
                    if _canon(c) == "publicfoodkey" or "public food key" in c.lower():
                        df = df.rename(columns={c: "Public Food Key"})
                        break

            if "Public Food Key" in df.columns:
                df["Public Food Key"] = df["Public Food Key"].astype(str).str.strip()

            return df
        except Exception:
            return None

    # -----------------------------
    # build final df
    # -----------------------------
    def _load_and_build(self) -> pd.DataFrame:
        nutr = self._load_nutrient_sheet()
        details = self._load_food_details()

        # Merge extra columns from food_details if present
        if details is not None and "Public Food Key" in details.columns:
            keep_cols = [c for c in details.columns if c not in nutr.columns]
            if keep_cols:
                nutr = nutr.merge(details[["Public Food Key"] + keep_cols], on="Public Food Key", how="left")

        # Convert all nutrient columns to numeric safely
        id_cols = {"Public Food Key", "Food Name", "Classification"}
        for c in nutr.columns:
            if c in id_cols:
                continue
            nutr[c] = pd.to_numeric(nutr[c], errors="coerce").fillna(0.0)

        # De-duplicate by key (keep first)
        nutr = nutr.drop_duplicates(subset=["Public Food Key"]).reset_index(drop=True)

        print(f"✔ NutrientService loaded: {len(nutr)} foods, {len(nutr.columns) - 3} nutrient columns")
        return nutr
