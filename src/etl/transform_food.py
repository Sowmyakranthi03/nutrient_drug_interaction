"""
transform_food.py

FSANZ-specific transformer.

Reads:
 - dataset/nutrient_databases/FSANZ/food_details.xlsx
 - dataset/nutrient_databases/FSANZ/measure_file.xlsx
 - dataset/nutrient_databases/FSANZ/nutrient_file.xlsx (may contain multiple sheets: solids & liquids)
 - dataset/nutrient_databases/FSANZ/recipe_file.xlsx
 - dataset/nutrient_databases/FSANZ/retention_factors.xlsx

Produces:
 - dataset/cleaned/foods_cleaned.csv  (one row per Public Food Key, nutrient_profile as JSON string)
 - dataset/cleaned/recipes_expanded.csv (recipes with computed nutrient_profile)
 - dataset/cleaned/nutrients_by_food.parquet (optional parquet with wide nutrient columns)

Notes:
 - Recipes: applies retention factors per-ingredient and weight-change % to compute final per-100g nutrient profile.
 - Liquids are marked is_liquid=True and nutrient values are per 100 mL (kept distinct); recipe computations assume ingredient units are in grams
"""
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

BASE = Path("dataset/nutrient_databases/FSANZ")
OUT_DIR = Path("dataset/cleaned")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _clean_colname(c: str) -> str:
    if not isinstance(c, str):
        return c
    return (
        c.strip()
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


def load_files():
    """Load all FSANZ files; returns dict of DataFrames."""
    food_file = BASE / "food_details.xlsx"
    nutrient_file = BASE / "nutrient_file.xlsx"  # may have multiple sheets
    measure_file = BASE / "measure_file.xlsx"
    recipe_file = BASE / "recipe_file.xlsx"
    retention_file = BASE / "retention_factors.xlsx"

    food_df = pd.read_excel(food_file, engine="openpyxl", dtype=str)
    # nutrient_file may have two sheets: solids and liquids (or tabs named)
    nut_xls = pd.read_excel(nutrient_file, sheet_name=None, engine="openpyxl")
    # flatten each sheet to dataframe
    # measure, recipe, retention
    measure_df = pd.read_excel(measure_file, engine="openpyxl", dtype=str)
    recipe_df = pd.read_excel(recipe_file, engine="openpyxl", dtype=str)
    retention_df = pd.read_excel(retention_file, engine="openpyxl", dtype=str)

    return {
        "food": food_df,
        "nutrients_sheets": nut_xls,
        "measures": measure_df,
        "recipes": recipe_df,
        "retention": retention_df,
    }


def normalize_header(df: pd.DataFrame) -> pd.DataFrame:
    """Strip, normalize column headers (but keep original readable names)."""
    df = df.copy()
    df.columns = [_clean_colname(c) for c in df.columns]
    return df


def detect_key_column(df: pd.DataFrame) -> str:
    """Return the name of the Public Food Key column as found in the DF."""
    candidates = [c for c in df.columns if c.lower().replace(" ", "").replace("_", "") in ("publicfoodkey", "key", "foodid", "food_key", "public_food_key", "key")]
    if candidates:
        return candidates[0]
    # fallbacks
    for c in df.columns:
        if "public" in c.lower() and "key" in c.lower():
            return c
    raise KeyError("Could not detect Public Food Key column. Please check file headers.")


def detect_name_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "food" in c.lower() and "name" in c.lower()]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if "name" == c.lower():
            return c
    raise KeyError("Could not detect Food Name column.")


def detect_specific_gravity_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "specific" in c.lower() and "gravity" in c.lower():
            return c
    return None


def build_nutrient_profiles_from_sheets(nut_sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Each nutrient sheet is assumed to be wide: first columns include Public Food Key, Classification, Food Name,
    then many nutrient columns with numeric values. Returns a combined DataFrame with:
      - public_food_key
      - food_name
      - classification (if present)
      - is_liquid (True for liquid sheets)
      - nutrient_* columns (kept as wide columns)
    """
    frames = []
    for sheet_name, df in nut_sheets.items():
        df = normalize_header(df)
        # detect key and name column
        try:
            key_col = detect_key_column(df)
        except KeyError:
            # try first column
            key_col = df.columns[0]
        try:
            name_col = detect_name_column(df)
        except KeyError:
            name_col = df.columns[1] if len(df.columns) > 1 else None

        # identify nutrient columns: exclude the first few metadata columns
        meta_cols = {key_col, name_col}
        # include classification if present
        class_cols = [c for c in df.columns if "class" in c.lower()]
        meta_cols.update(class_cols)
        # any non-numeric columns among the rest are still nutrients (but will be coerced)
        nutrient_cols = [c for c in df.columns if c not in meta_cols]

        # coerce nutrients to numeric where possible
        for c in nutrient_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        subset = df[[key_col, name_col] + class_cols + nutrient_cols].copy()
        subset = subset.rename(columns={key_col: "public_food_key", name_col: "food_name"})
        subset["is_liquid"] = True if ("liquid" in sheet_name.lower() or "per 100 ml" in sheet_name.lower()) else False
        frames.append(subset)

    # Combine solids & liquids; if same public_food_key exists in both, we keep them as separate rows with is_liquid flag.
    combined = pd.concat(frames, ignore_index=True, sort=False)
    # drop rows without public_food_key
    combined = combined.dropna(subset=["public_food_key"])
    combined["public_food_key"] = combined["public_food_key"].astype(str).str.strip()
    return combined


def load_retention_factors(retention_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean retention factors DF. Columns include:
      Retention Factor ID, Retention Factor Description, Original USDA ...
      then many nutrient columns (Alpha carotene, Calcium (Ca), Vitamin C, etc.)
    We'll convert nutrient column names into normalized keys.
    """
    df = normalize_header(retention_df)
    # detect id column
    id_col = None
    for c in df.columns:
        if "retention" in c.lower() and ("id" in c.lower() or "factor id" in c.lower() or "id" == c.lower()):
            id_col = c
            break
    if not id_col:
        # fallback to first column
        id_col = df.columns[0]

    # everything except id and description are nutrient retention factors
    descr_col = None
    for c in df.columns:
        if "description" in c.lower():
            descr_col = c
            break

    nutrient_cols = [c for c in df.columns if c not in {id_col, descr_col}]
    # coerce numeric
    for c in nutrient_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={id_col: "retention_factor_id"})
    df["retention_factor_id"] = df["retention_factor_id"].astype(str).str.strip()
    return df.set_index("retention_factor_id")


def build_food_nutrient_profiles(nut_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by public_food_key (with possibly duplicate rows for solids/liquids).
    Also produces a 'nutrient_profile' column (dict) serialized as JSON string.
    """
    # nutrient columns are those other than metadata
    meta = ["public_food_key", "food_name", "classification", "is_liquid"]
    meta = [c for c in meta if c in nut_combined.columns]
    nutrient_cols = [c for c in nut_combined.columns if c not in meta]

    records = []
    for _, row in nut_combined.iterrows():
        pk = str(row["public_food_key"])
        profile = {}
        for c in nutrient_cols:
            val = row.get(c)
            if pd.notnull(val):
                # store numeric value (per 100g or per 100 mL depending on is_liquid)
                try:
                    f = float(val)
                except Exception:
                    continue
                # Normalize column name: remove units in parens and nasty whitespace
                normalized = c.strip()
                profile[normalized] = f
        rec = {
            "public_food_key": pk,
            "food_name": row.get("food_name"),
            "classification": row.get("classification") if "classification" in row else None,
            "is_liquid": bool(row.get("is_liquid", False)),
            "nutrient_profile": profile,
        }
        records.append(rec)

    df = pd.DataFrame(records)
    return df


def apply_recipe_processing(recipes_df: pd.DataFrame, foods_df: pd.DataFrame, retention_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each recipe row (which contains ingredient entries), compute nutrient profile for final recipe.

    Input assumptions:
     - recipes_df contains for each recipe: Public Food Key (recipe key), Total Weight Change (%), Ingredient Public Food Key, Ingredient Weight (g), Retention Factor ID (optional)
     - foods_df has index public_food_key and a nutrient_profile dict per food (per 100g).
    """
    recipes_df = normalize_header(recipes_df)

    # detect columns
    key_col = detect_key_column(recipes_df)
    recipe_name_col = None
    for c in recipes_df.columns:
        if "food" in c.lower() and "name" in c.lower():
            recipe_name_col = c
            break
    weight_change_col = None
    for c in recipes_df.columns:
        if "total" in c.lower() and "weight" in c.lower():
            weight_change_col = c
            break
    ingredient_key_col = None
    for c in recipes_df.columns:
        if "ingredient" in c.lower() and "public" in c.lower() and "key" in c.lower():
            ingredient_key_col = c
            break
    ingredient_weight_col = None
    for c in recipes_df.columns:
        if "ingredient" in c.lower() and "weight" in c.lower():
            ingredient_weight_col = c
            break
    retention_id_col = None
    for c in recipes_df.columns:
        if "retention" in c.lower() and "id" in c.lower():
            retention_id_col = c
            break

    if ingredient_key_col is None or ingredient_weight_col is None:
        raise KeyError("Cannot detect ingredient key/weight columns in recipe file.")

    # group recipes by recipe key
    grouped = recipes_df.groupby(key_col)
    recipe_records = []
    for recipe_key, group in grouped:
        recipe_key = str(recipe_key)
        recipe_name = group[recipe_name_col].iloc[0] if recipe_name_col in group.columns else None
        total_weight_raw = group[ingredient_weight_col].apply(pd.to_numeric, errors="coerce").sum()
        weight_change_pct = 0.0
        if weight_change_col and weight_change_col in group.columns:
            # the same value is repeated per row, pick first nonnull
            try:
                weight_change_pct = float(group[weight_change_col].dropna().iloc[0])
            except Exception:
                weight_change_pct = 0.0
        weight_multiplier = 1.0 + (weight_change_pct / 100.0)
        final_weight = total_weight_raw * weight_multiplier if total_weight_raw and not math.isclose(total_weight_raw, 0.0) else 0.0

        # accumulate nutrient amounts (absolute amounts in mg or g consistent with source units)
        # for each ingredient:
        total_nutrients = {}  # nutrient_name -> total_amount_in_recipe (same units as per-100g * g factor)
        for _, ing in group.iterrows():
            ing_key = str(ing[ingredient_key_col]).strip()
            try:
                ing_weight = float(ing[ingredient_weight_col])
            except Exception:
                ing_weight = 0.0
            retention_id = None
            if retention_id_col and retention_id_col in ing.index:
                retention_id = str(ing[retention_id_col]) if pd.notnull(ing[retention_id_col]) else None

            # fetch ingredient nutrient profile (per 100 g)
            frow = foods_df[foods_df["public_food_key"] == ing_key]
            if frow.empty:
                # ingredient not found; skip
                # optionally log missing ingredient
                # print(f"[WARN] Ingredient {ing_key} not found in foods table.")
                continue
            ing_profile = frow.iloc[0]["nutrient_profile"] or {}
            # for each nutrient in ingredient, compute amount contributed: (ing_weight / 100) * value_per100g * retention_factor
            for nut_name, val_per100 in ing_profile.items():
                # find retention factor for this nutrient (if retention_id exists); retention_df indexed by id; columns are nutrient names
                rf = 1.0
                if retention_id and retention_id in retention_df.index:
                    # retention_df may have nutrient columns with slightly different names -> try direct lookup and case-insensitive fallback
                    retention_row = retention_df.loc[retention_id]
                    # direct
                    if nut_name in retention_row.index:
                        rf_val = retention_row.get(nut_name)
                        if pd.notnull(rf_val):
                            try:
                                rf = float(rf_val)
                            except Exception:
                                rf = 1.0
                    else:
                        # try case-insensitive match
                        lc_map = {c.lower(): c for c in retention_row.index}
                        if nut_name.lower() in lc_map:
                            rf_val = retention_row.get(lc_map[nut_name.lower()])
                            if pd.notnull(rf_val):
                                try:
                                    rf = float(rf_val)
                                except Exception:
                                    rf = 1.0
                # contribution amount (in same units as val_per100): value_per100 * (ing_weight/100) * rf
                contrib = 0.0
                try:
                    contrib = float(val_per100) * (ing_weight / 100.0) * rf
                except Exception:
                    contrib = 0.0
                total_nutrients[nut_name] = total_nutrients.get(nut_name, 0.0) + contrib

        # compute per-100g of final product (if final_weight > 0)
        per100_profile = {}
        if final_weight and final_weight > 0:
            for nut_name, total_amt in total_nutrients.items():
                # amount per 100g = (total_amt / final_weight) * 100
                per100 = (total_amt / final_weight) * 100.0
                per100_profile[nut_name] = per100
        else:
            # fallback: divide by raw total weight
            if total_weight_raw and total_weight_raw > 0:
                for nut_name, total_amt in total_nutrients.items():
                    per100_profile[nut_name] = (total_amt / total_weight_raw) * 100.0
            else:
                per100_profile = {}

        recipe_records.append({
            "recipe_public_food_key": str(recipe_key),
            "recipe_name": recipe_name,
            "total_weight_raw_g": total_weight_raw,
            "weight_change_pct": weight_change_pct,
            "final_weight_g": final_weight,
            "nutrient_profile": per100_profile
        })

    recipes_expanded_df = pd.DataFrame(recipe_records)
    return recipes_expanded_df


def main():
    print("Loading files...")
    data = load_files()
    food_df = normalize_header(data["food"])
    measures_df = normalize_header(data["measures"])
    recipes_df = data["recipes"]  # normalized later inside recipe function
    retention_df = load_retention_factors(data["retention"])

    print("Building nutrient table from nutrient file sheets...")
    nut_combined = build_nutrient_profiles_from_sheets({k: normalize_header(v) for k, v in data["nutrients_sheets"].items()})

    print("Constructing foods dataframe with nutrient_profile dicts...")
    foods_nutr = build_food_nutrient_profiles(nut_combined)

    # merge with food details to get classification / specific gravity if present
    key_col_food = detect_key_column(food_df)
    name_col_food = detect_name_column(food_df)
    sg_col = detect_specific_gravity_col(food_df)
    food_df[key_col_food] = food_df[key_col_food].astype(str).str.strip()

    foods_nutr = foods_nutr.merge(
        food_df[[key_col_food, name_col_food] + ([sg_col] if sg_col else [])].rename(
            columns={key_col_food: "public_food_key", name_col_food: "food_name", sg_col: "specific_gravity" if sg_col else "specific_gravity"}
        ),
        on="public_food_key",
        how="left"
    )

    # unify name preference
    foods_nutr["food_name"] = foods_nutr["food_name_x"].fillna(foods_nutr["food_name_y"]) if "food_name_x" in foods_nutr.columns else foods_nutr["food_name"]

    # Apply recipe expansion (compute per-100g nutrient profiles for recipes where ingredients are listed)
    print("Computing recipe nutrient profiles using retention factors (if recipe file present)...")
    recipes_expanded = apply_recipe_processing(recipes_df, foods_nutr, retention_df)

    # Append recipe-derived foods to foods_nutr (if a recipe_public_food_key does not exist as a food already)
    recipes_expanded = recipes_expanded.dropna(subset=["recipe_public_food_key"])
    recipes_expanded["public_food_key"] = recipes_expanded["recipe_public_food_key"]
    recipes_expanded["food_name"] = recipes_expanded["recipe_name"]
    recipes_expanded["classification"] = "recipe"
    recipes_expanded["is_liquid"] = False
    recipes_expanded = recipes_expanded[["public_food_key", "food_name", "classification", "is_liquid", "nutrient_profile"]]

    # Ensure nutrient_profile column is JSON serializable (convert numpy floats)
    def norm_profile(p):
        if not isinstance(p, dict):
            return {}
        out = {}
        for k, v in p.items():
            try:
                out[k] = float(v)
            except Exception:
                out[k] = None
        return out

    foods_nutr["nutrient_profile"] = foods_nutr["nutrient_profile"].apply(norm_profile)
    recipes_expanded["nutrient_profile"] = recipes_expanded["nutrient_profile"].apply(norm_profile)

    # Merge: if recipe key already exists, overwrite or keep both? We'll upsert recipe if not present.
    # Build final foods table: existing foods + recipes not present
    existing_keys = set(foods_nutr["public_food_key"].astype(str).tolist())
    recipes_to_add = recipes_expanded[~recipes_expanded["public_food_key"].isin(existing_keys)].copy()

    final_foods = pd.concat([
        foods_nutr[["public_food_key", "food_name", "classification", "is_liquid", "nutrient_profile"]],
        recipes_to_add[["public_food_key", "food_name", "classification", "is_liquid", "nutrient_profile"]]
    ], ignore_index=True, sort=False)

    # Attach measure gram weights (if present) - match on Public Food Key
    measure_key_col = detect_key_column(measures_df)
    measures_df[measure_key_col] = measures_df[measure_key_col].astype(str).str.strip()
    measures_small = measures_df[[measure_key_col, "weight in grams"]].copy() if "weight in grams" in measures_df.columns else measures_df[[measure_key_col]].copy()
    measures_small = measures_small.rename(columns={measure_key_col: "public_food_key", "weight in grams": "gram_weight"})
    final_foods = final_foods.merge(measures_small, on="public_food_key", how="left")

    # Save outputs
    print("Saving foods_cleaned.csv ...")
    # write nutrient_profile as JSON string for CSV friendliness
    out_df = final_foods.copy()
    out_df["nutrient_profile"] = out_df["nutrient_profile"].apply(lambda d: json.dumps(d, ensure_ascii=False))
    out_df.to_csv(OUT_DIR / "foods_cleaned.csv", index=False)

    print("Saving recipes_expanded.csv ...")
    recipes_expanded["nutrient_profile"] = recipes_expanded["nutrient_profile"].apply(lambda d: json.dumps(d, ensure_ascii=False))
    recipes_expanded.to_csv(OUT_DIR / "recipes_expanded.csv", index=False)

    # optional: save wide Parquet (explode nutrient_profile into columns) - only do if small enough
    try:
        print("Saving nutrients_by_food.parquet (wide) ...")
        # create wide DF (careful: many columns)
        wide_rows = []
        for _, r in final_foods.iterrows():
            base = {
                "public_food_key": r["public_food_key"],
                "food_name": r["food_name"],
                "classification": r.get("classification"),
                "is_liquid": r.get("is_liquid"),
            }
            prof = r["nutrient_profile"] or {}
            for k, v in prof.items():
                # normalize column name
                col = str(k).strip()
                base[col] = v
            wide_rows.append(base)
        wide_df = pd.DataFrame(wide_rows)
        wide_df.to_parquet(OUT_DIR / "nutrients_by_food.parquet", index=False)
    except Exception as e:
        print("Warning: could not write wide parquet (too many columns or memory).", e)

    print("All done. Outputs in", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
