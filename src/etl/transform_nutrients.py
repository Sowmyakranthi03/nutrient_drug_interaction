"""
transform_nutrients.py

Reads FSANZ nutrient_file.xlsx (solids per 100 g & liquids per 100 mL),
normalizes headers, extracts nutrients, and outputs:
 - foods_nutrients_wide.csv
 - foods_nutrients_long.csv
 - foods_nutrients_profiles.json
"""

import re
import json
from pathlib import Path
import pandas as pd

# -----------------------------
# CONFIG (update paths here)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
FSANZ_NUTRIENT_FILE = PROJECT_ROOT / "dataset/nutrient_databases/FSANZ/nutrient_file.xlsx"
OUT_DIR = PROJECT_ROOT / "dataset/cleaned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def normalize_header(col: str) -> str:
    """Clean header string into safe column key."""
    if not isinstance(col, str):
        return str(col)
    s = col.replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    s = s.replace("%", "percent")
    s = re.sub(r"[()]", " ", s)
    s = re.sub(r"[^0-9a-zA-Z_ ]+", "", s)
    s = s.strip().replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s

def detect_key_and_name(df: pd.DataFrame):
    """Return (key_col, name_col) detected from df columns."""
    cols = list(df.columns)
    key_candidates = [c for c in cols if re.search(r"(public.*food.*key|public_food_key|publicfoodkey|key|food_key|foodid|public_key)", c, re.I)]
    name_candidates = [c for c in cols if re.search(r"(food.*name|food_name|name)", c, re.I)]
    key = key_candidates[0] if key_candidates else cols[0]
    name = name_candidates[0] if name_candidates else (cols[1] if len(cols) > 1 else None)
    return key, name

def split_unit_from_header(raw_header: str):
    """Extract unit from header parentheses."""
    if not isinstance(raw_header, str):
        return raw_header, None
    m = re.findall(r"\((.*?)\)", raw_header)
    unit = None
    if m:
        last = m[-1].lower()
        if any(x in last for x in ["mg", "g", "ug", "kj", "kcal", "ml", "m l", "percent", "µg"]):
            unit = last.replace(" ", "")
    key = normalize_header(raw_header)
    return key, unit

# -----------------------------
# MAIN TRANSFORM FUNCTION
# -----------------------------
def transform_nutrients(nutrient_file: Path):
    print(f"Loading nutrient file: {nutrient_file}")
    xls = pd.read_excel(nutrient_file, sheet_name=None, engine="openpyxl")

    per_food_profiles = {}
    long_rows = []
    wide_rows = []

    for sheet_name, df in xls.items():
        print(f"Processing sheet: {sheet_name} (shape={df.shape})")
        # normalize headers & map units
        raw_cols = list(df.columns)
        col_unit_map = {}
        normalized_cols = []
        for c in raw_cols:
            nk, unit = split_unit_from_header(str(c))
            normalized_cols.append(nk)
            col_unit_map[nk] = unit
        df.columns = normalized_cols

        key_col, name_col = detect_key_and_name(df)
        is_liquid = bool(re.search(r"liquid|100\s*ml|100ml", sheet_name, re.I))

        for _, row in df.iterrows():
            pkey = str(row.get(key_col)).strip() if pd.notna(row.get(key_col)) else None
            fname = row.get(name_col) if name_col and name_col in row.index else None
            if pkey is None or pkey == "nan":
                continue

            profile = {}
            wide_entry = {"public_food_key": pkey, "food_name": fname, "sheet": sheet_name, "is_liquid_sheet": is_liquid}

            for col in df.columns:
                if col in [key_col, name_col]:
                    continue
                val = row.get(col)
                if pd.isna(val):
                    continue
                try:
                    valf = float(val)
                except Exception:
                    continue
                profile[col] = valf
                wide_entry[col] = valf
                long_rows.append({
                    "public_food_key": pkey,
                    "food_name": fname,
                    "nutrient": col,
                    "value": valf,
                    "unit": col_unit_map.get(col),
                    "is_liquid": is_liquid
                })

            if pkey in per_food_profiles:
                existing = per_food_profiles[pkey]
                if isinstance(existing, dict) and "_variants" not in existing:
                    per_food_profiles[pkey] = {"_variants": [existing, profile]}
                else:
                    per_food_profiles[pkey]["_variants"].append(profile)
            else:
                per_food_profiles[pkey] = profile

            wide_rows.append(wide_entry)

    # Wide dataframe
    wide_df = pd.DataFrame(wide_rows).sort_values(by=["public_food_key", "is_liquid_sheet"]).drop_duplicates(subset=["public_food_key", "is_liquid_sheet"], keep="first")
    long_df = pd.DataFrame(long_rows)

    # Canonical profiles
    canonical_profiles = {}
    for pkey, profile_obj in per_food_profiles.items():
        if isinstance(profile_obj, dict) and "_variants" in profile_obj:
            variants = profile_obj["_variants"]
            selected = max(variants, key=lambda d: len(d.keys()))
            canonical_profiles[pkey] = selected
        elif isinstance(profile_obj, dict):
            canonical_profiles[pkey] = profile_obj
        else:
            canonical_profiles[pkey] = {}

    # Save outputs
    wide_df.to_csv(OUT_DIR / "foods_nutrients_wide.csv", index=False, encoding="utf-8")
    long_df.to_csv(OUT_DIR / "foods_nutrients_long.csv", index=False, encoding="utf-8")
    with open(OUT_DIR / "foods_nutrients_profiles.json", "w", encoding="utf-8") as f:
        serializable_profiles = {k: {nk: float(nv) for nk, nv in v.items()} for k, v in canonical_profiles.items()}
        json.dump(serializable_profiles, f, ensure_ascii=False, indent=2)

    print(f"Saved wide CSV ({wide_df.shape[0]} foods), long CSV ({long_df.shape[0]} rows), JSON profiles ({len(canonical_profiles)} foods).")
    return wide_df, long_df, canonical_profiles

# -----------------------------
# CLI / RUN
# -----------------------------
if __name__ == "__main__":
    if not FSANZ_NUTRIENT_FILE.exists():
        raise FileNotFoundError(f"FSANZ nutrient file not found at {FSANZ_NUTRIENT_FILE}")
    transform_nutrients(FSANZ_NUTRIENT_FILE)
