from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional
from src.config.settings import FSANZ_NUTRIENT_XLSX, FSANZ_FOOD_DETAILS_XLSX

def _norm_col(c: str) -> str:
    c = str(c).replace("\n", " ").replace("\r", " ").strip()
    c = " ".join(c.split())
    return c

class NutrientService:
    def __init__(self, nutrient_path=None, food_details_path=None):
        self.nutrient_path = nutrient_path or FSANZ_NUTRIENT_XLSX
        self.food_details_path = food_details_path or FSANZ_FOOD_DETAILS_XLSX
        self.df = self._load_and_build()

    def _pick_sheet(self, xls: pd.ExcelFile) -> str:
        # Your file shows: "All solids & liquids per 100g" and "Liquids only per 100mL"
        # We want the 100g one.
        candidates = xls.sheet_names
        for s in candidates:
            if "100g" in s.lower() or "100 g" in s.lower():
                return s
        # fallback: first sheet
        return candidates[0]

    def _load_nutrient_sheet(self) -> pd.DataFrame:
        xls = pd.ExcelFile(self.nutrient_path)
        sheet = self._pick_sheet(xls)
        df = pd.read_excel(self.nutrient_path, sheet_name=sheet)

        # Normalize columns
        df.columns = [_norm_col(c) for c in df.columns]

        # Fix common naming variations
        col_map = {}
        for c in df.columns:
            low = c.lower()
            if low == "food name" or low == "food name " or low == "food name.":
                col_map[c] = "Food Name"
            if low == "food name" or low == "food name":
                col_map[c] = "Food Name"
            if low == "food name" or low == "food name":
                col_map[c] = "Food Name"

            if low == "food name" or low == "food name":
                col_map[c] = "Food Name"

            if low == "food name" or low == "food name":
                col_map[c] = "Food Name"

            if low == "food name" or low == "food name":
                col_map[c] = "Food Name"

        # Real variants seen in your screenshots:
        for c in df.columns:
            if c.lower() == "food name":
                col_map[c] = "Food Name"
            if c.lower() == "food name ":
                col_map[c] = "Food Name"
            if c.lower() == "food name":
                col_map[c] = "Food Name"
            if c.lower() == "food name":
                col_map[c] = "Food Name"
            if c.lower() == "food name":
                col_map[c] = "Food Name"
            if c.lower() == "food name":
                col_map[c] = "Food Name"

        # also handle "Food name"
        for c in df.columns:
            if c.lower() == "food name" or c.lower() == "food name":
                col_map[c] = "Food Name"

        # handle exact you showed: "Food name"
        for c in df.columns:
            if c == "Food name":
                col_map[c] = "Food Name"

        df = df.rename(columns=col_map)

        # Ensure required columns exist
        if "Public Food Key" not in df.columns:
            # sometimes it becomes "Public Food Key " or has weird spaces
            for c in df.columns:
                if c.lower().replace(" ", "") == "publicfoodkey":
                    df = df.rename(columns={c: "Public Food Key"})
                    break

        if "Food Name" not in df.columns:
            # sometimes "Food Name" already exists; else try fuzzy fallback
            for c in df.columns:
                if c.lower().replace(" ", "") in ["foodname", "food_name"]:
                    df = df.rename(columns={c: "Food Name"})
                    break

        if "Classification" not in df.columns:
            for c in df.columns:
                if c.lower().replace(" ", "") == "classification":
                    df = df.rename(columns={c: "Classification"})
                    break

        for must in ["Public Food Key", "Food Name", "Classification"]:
            if must not in df.columns:
                raise ValueError(f"Missing required column: {must} | found={list(df.columns)[:20]}...")

        # Clean key + names
        df["Public Food Key"] = df["Public Food Key"].astype(str).str.strip()
        df["Food Name"] = df["Food Name"].astype(str).str.strip()

        # Drop empty/nan names
        df = df[df["Food Name"].notna()]
        df = df[df["Food Name"].str.lower() != "nan"]
        df = df[df["Food Name"].str.len() > 0]

        return df

    def _load_food_details(self) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_excel(self.food_details_path)
            df.columns = [_norm_col(c) for c in df.columns]
            # normalize key
            if "Public Food Key" not in df.columns:
                for c in df.columns:
                    if c.lower().replace(" ", "") == "publicfoodkey":
                        df = df.rename(columns={c: "Public Food Key"})
                        break
            if "Food Name" not in df.columns:
                for c in df.columns:
                    if c.lower() in ["food name", "foodname", "name"]:
                        df = df.rename(columns={c: "Food Name"})
                        break
            if "Public Food Key" in df.columns:
                df["Public Food Key"] = df["Public Food Key"].astype(str).str.strip()
            return df
        except Exception:
            return None

    def _load_and_build(self) -> pd.DataFrame:
        nutr = self._load_nutrient_sheet()
        details = self._load_food_details()

        if details is not None and "Public Food Key" in details.columns:
            # join extra columns if needed
            keep_cols = [c for c in details.columns if c not in nutr.columns]
            if keep_cols:
                nutr = nutr.merge(details[["Public Food Key"] + keep_cols], on="Public Food Key", how="left")

        # Convert numeric columns safely
        id_cols = {"Public Food Key", "Food Name", "Classification"}
        for c in nutr.columns:
            if c in id_cols:
                continue
            nutr[c] = pd.to_numeric(nutr[c], errors="coerce").fillna(0.0)

        print(f"✔ NutrientService loaded: {len(nutr)} foods, {len(nutr.columns) - 3} nutrient columns")
        return nutr
