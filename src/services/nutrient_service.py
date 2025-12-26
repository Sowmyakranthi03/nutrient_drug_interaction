# src/services/nutrient_service.py
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np


class NutrientService:
    def __init__(self):
        # Update this path to your actual file location
        self.nutrient_path = Path("dataset/nutrient_databases/FSANZ/nutrient_file.xlsx")

        self.df = self._load_wide_fsanz()
        print(f"✔ NutrientService loaded: {len(self.df)} foods, {len(self.df.columns) - 3} nutrient columns")

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).strip().lower())

    def _pick_sheet(self, xls: pd.ExcelFile) -> str:
        """Pick the 100g sheet reliably."""
        sheets = xls.sheet_names
        # Prefer “All solids & liquids per 100g”
        for s in sheets:
            if "100g" in self._norm(s) and "solids" in self._norm(s):
                return s
        # fallback: any sheet with 100g
        for s in sheets:
            if "100g" in self._norm(s):
                return s
        # last resort: first sheet
        return sheets[0]

    @staticmethod
    def _coerce_numeric(series: pd.Series) -> pd.Series:
        # remove commas in numbers "1,236" -> "1236"
        s = series.astype(str).str.replace(",", "", regex=False)
        return pd.to_numeric(s, errors="coerce")

    # ----------------------------
    # Main loader
    # ----------------------------
    def _load_wide_fsanz(self) -> pd.DataFrame:
        if not self.nutrient_path.exists():
            raise FileNotFoundError(f"FSANZ nutrient file not found: {self.nutrient_path}")

        xls = pd.ExcelFile(self.nutrient_path)
        sheet = self._pick_sheet(xls)

        df = pd.read_excel(self.nutrient_path, sheet_name=sheet, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]

        # --- Normalize required columns (some files use "Food name")
        col_map = {}
        lower_map = {self._norm(c): c for c in df.columns}

        def rename_if_exists(want: str, canonical: str):
            if want in lower_map:
                col_map[lower_map[want]] = canonical

        rename_if_exists("public food key", "Public Food Key")
        rename_if_exists("food name", "Food Name")
        rename_if_exists("food name ", "Food Name")
        rename_if_exists("food name\r\n", "Food Name")
        rename_if_exists("classification", "Classification")

        df = df.rename(columns=col_map)

        # --- Some FSANZ exports include junk first column like "Unnamed: 0"
        drop_cols = [c for c in df.columns if self._norm(c).startswith("unnamed")]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # --- Required columns check
        required = ["Public Food Key", "Food Name", "Classification"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}. Found: {list(df.columns)[:10]} ...")

        # --- Drop “units row / header row inside data”
        # Keep only rows where Public Food Key looks like 'F000123' style
        key = df["Public Food Key"].astype(str).str.strip()
        df = df[key.str.match(r"^F\d+", na=False)].copy()

        # --- Clean food name
        df["Food Name"] = df["Food Name"].astype(str).str.strip()
        df = df[df["Food Name"].notna() & (df["Food Name"] != "")]

        # --- Convert nutrient columns to numeric safely
        id_cols = {"Public Food Key", "Food Name", "Classification"}
        nutrient_cols = [c for c in df.columns if c not in id_cols]

        for c in nutrient_cols:
            df[c] = self._coerce_numeric(df[c]).fillna(0.0)

        # --- Remove duplicates by key (keep first)
        df["Public Food Key"] = df["Public Food Key"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["Public Food Key"], keep="first").reset_index(drop=True)

        return df
