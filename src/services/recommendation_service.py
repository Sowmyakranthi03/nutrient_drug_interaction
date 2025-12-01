# src/services/recommendation_service.py

from __future__ import annotations

from typing import List, Dict, Any, Set
import pandas as pd

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.services.interaction_service import InteractionService
from src.services.nutrient_service import NutrientService


# Keywords that we consider in drug food_interaction texts and in food names
FOOD_KEYWORDS = [
    # Herbs / botanicals
    "garlic",
    "ginger",
    "ginseng",
    "ginkgo",
    "ginkgo biloba",
    "chamomile",
    "echinacea",
    "bilberry",
    "danshen",
    "piracetam",
    "st. john",
    "st john",

    # Common food interactions
    "grapefruit",
    "alcohol",
    "wine",
    "beer",

    # Dairy
    "milk",
    "dairy",
    "cheese",
    "yogurt",

    # Caffeine / drinks
    "coffee",
    "tea",
    "caffeine",

    # Vitamins
    "vitamin k",
    "vitamin c",

    # Meal patterns
    "high-fat",
    "high fat",
]

# Nutrient-based rules per drug, matched against FSANZ column names.
DRUG_NUTRIENT_LIMITS: Dict[str, List[Dict[str, Any]]] = {
    # Example: Warfarin → limit Vitamin K intake
    "warfarin": [
        {"nutrient_substr": "vitamin k", "max_per_100g": 30.0},
    ],
}


class RecommendationService:
    """
    Random Forest–based recommendation service without PKL files.

    Combines:
      - Global ML model (nutrient profiles → safe/unsafe).
      - Name-based conflicts from DrugBank (garlic, ginger, grapefruit, etc.).
      - Drug-specific nutrient limits (e.g. Warfarin → Vitamin K).
      - Drug-specific preferences (e.g. Peginterferon alfa-2a → high-water foods).
    """

    def __init__(self):
        self.interaction_service = InteractionService()
        self.nutrient_service = NutrientService()

        # FSANZ nutrient dataframe from NutrientService
        self.df = self.nutrient_service.df

        # Build all internal structures and train model
        self._prepare_training_data()
        self._build_nutrient_limit_index()
        self._train_model()

        print(
            f"✔ RecommendationService ready — "
            f"{len(self.food_keys)} foods, "
            f"{len(self.feature_order)} nutrient features."
        )

    # ------------------------------------------------------------------
    # DATA PREPARATION
    # ------------------------------------------------------------------
    def _prepare_training_data(self):
        df = self.df

        # Basic identifiers
        self.food_keys: List[str] = df["Public Food Key"].astype(str).tolist()
        self.food_names: List[str] = df["Food Name"].astype(str).tolist()

        # Nutrient columns used as ML features
        id_cols = {"Public Food Key", "Food Name", "Classification"}
        self.feature_order: List[str] = [c for c in df.columns if c not in id_cols]

        print(f"✔ Nutrient columns used for ML: {len(self.feature_order)}")

        # Build feature matrix (foods × nutrients)
        feature_rows = []
        for _, row in df.iterrows():
            vals = []
            for col in self.feature_order:
                v = row.get(col)
                try:
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        vals.append(0.0)
                    else:
                        vals.append(float(v))
                except Exception:
                    vals.append(0.0)
            feature_rows.append(vals)

        self.feature_matrix = np.array(feature_rows, dtype=float)

        # Detect water/moisture column for hydration scoring (Peginterferon)
        self.water_col_name = None
        for col in self.df.columns:
            # Handles "Moisture (water) (g)" or similar
            if "moisture (water" in col.lower():
                self.water_col_name = col
                break

        if self.water_col_name:
            # SAFE conversion: coerce non-numeric (e.g. 'per 100 mL') to NaN, then 0
            water_series = (
                self.df[self.water_col_name]
                .fillna(0.0)
            )
            water_numeric = (
                pd.to_numeric(water_series, errors="coerce")
                .fillna(0.0)
            )
            self.water_values = water_numeric.to_numpy(dtype=float)
            self.max_water = float(self.water_values.max()) if self.water_values.size > 0 else 0.0
            print(f"✔ Using '{self.water_col_name}' for hydration scoring")
        else:
            self.water_values = None
            self.max_water = 0.0
            print("⚠ No water/moisture column found for hydration scoring")

        # Discover global bad keywords from all DrugBank interactions
        self.global_bad_keywords: Set[str] = self._discover_global_keywords()
        print(f"✔ Global bad keywords from DrugBank: {self.global_bad_keywords}")

        # Labels: unsafe (1) if food name contains any global bad keyword, else safe (0)
        labels = []
        for name in self.food_names:
            nl = name.lower()
            unsafe = any(k in nl for k in self.global_bad_keywords)
            labels.append(int(unsafe))

        self.labels = np.array(labels, dtype=int)
        unique, counts = np.unique(self.labels, return_counts=True)
        print("✔ Label distribution:", dict(zip(unique, counts)))

    # ------------------------------------------------------------------
    # BUILD NUTRIENT LIMIT INDEX
    # ------------------------------------------------------------------
    def _build_nutrient_limit_index(self):
        """
        Precompute which FSANZ columns correspond to each nutrient_substr
        used in DRUG_NUTRIENT_LIMITS.
        """
        self.nutrient_limit_columns: Dict[str, List[str]] = {}
        all_substrings: Set[str] = set()

        for rules in DRUG_NUTRIENT_LIMITS.values():
            for r in rules:
                all_substrings.add(r["nutrient_substr"].lower())

        for substr in all_substrings:
            cols = [c for c in self.df.columns if substr in c.lower()]
            if cols:
                self.nutrient_limit_columns[substr] = cols

        if self.nutrient_limit_columns:
            print("✔ Nutrient limit column index:")
            for k, v in self.nutrient_limit_columns.items():
                print(f"   '{k}' → {v}")

    # ------------------------------------------------------------------
    # GLOBAL KEYWORD DISCOVERY
    # ------------------------------------------------------------------
    def _discover_global_keywords(self) -> Set[str]:
        bad_keywords: Set[str] = set()
        for interactions in self.interaction_service.drug_map.values():
            for txt in interactions:
                lower = txt.lower()
                for kw in FOOD_KEYWORDS:
                    if kw in lower:
                        bad_keywords.add(kw)
        return bad_keywords

    # ------------------------------------------------------------------
    # MODEL TRAINING
    # ------------------------------------------------------------------
    def _train_model(self):
        if len(np.unique(self.labels)) == 1:
            print("⚠ WARNING: All labels are identical. Model will be degenerate.")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.feature_matrix)

        self.model = RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, self.labels)
        print("✔ RandomForest model trained")

    # ------------------------------------------------------------------
    # LOW-LEVEL PREDICTIONS
    # ------------------------------------------------------------------
    def _predict_food_safety_by_index(self, idx: int) -> bool:
        row = self.feature_matrix[idx]
        scaled = self.scaler.transform([row])
        pred = self.model.predict(scaled)[0]
        return bool(pred == 0)  # 0 = safe

    def _predict_proba_safe_by_index(self, idx: int) -> float:
        row = self.feature_matrix[idx]
        scaled = self.scaler.transform([row])

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(scaled)[0]
            return float(proba[0])  # class 0 = safe
        pred = self.model.predict(scaled)[0]
        return 1.0 if pred == 0 else 0.0

    # ------------------------------------------------------------------
    # PER-DRUG BAD TERMS
    # ------------------------------------------------------------------
    def _bad_terms_from_drugs(self, drug_list) -> Set[str]:
        if isinstance(drug_list, str):
            drug_list = [drug_list]

        bad_terms: Set[str] = set()
        for drug in drug_list:
            interactions = self.interaction_service.get_drug_interactions(drug)
            for txt in interactions:
                lower = txt.lower()
                for kw in FOOD_KEYWORDS:
                    if kw in lower:
                        bad_terms.add(kw)

        print(f"✔ Per-drug bad terms for {drug_list}: {bad_terms}")
        return bad_terms

    # ------------------------------------------------------------------
    # PER-DRUG NUTRIENT LIMITS
    # ------------------------------------------------------------------
    def _combined_drug_limits(self, drug_list) -> List[Dict[str, Any]]:
        """
        Combine nutrient limits from all drugs in drug_list.
        If same nutrient_substr appears multiple times, take the minimum max_per_100g.
        """
        if isinstance(drug_list, str):
            drug_list = [drug_list]

        limits: Dict[str, float] = {}

        for drug in drug_list:
            dname = drug.lower().strip()
            rules = DRUG_NUTRIENT_LIMITS.get(dname)
            if not rules:
                continue
            for r in rules:
                key = r["nutrient_substr"].lower()
                max_val = float(r["max_per_100g"])
                if key in limits:
                    limits[key] = min(limits[key], max_val)
                else:
                    limits[key] = max_val

        combined = [
            {"nutrient_substr": k, "max_per_100g": v}
            for k, v in limits.items()
        ]

        if combined:
            print(f"✔ Nutrient limits for {drug_list}: {combined}")
        return combined

    def _violates_nutrient_limits(self, food_index: int, limits: List[Dict[str, Any]]) -> bool:
        """
        Return True if the food at food_index exceeds any nutrient limit.
        """
        if not limits:
            return False

        row = self.df.iloc[food_index]

        for rule in limits:
            substr = rule["nutrient_substr"].lower()
            max_val = rule["max_per_100g"]

            cols = self.nutrient_limit_columns.get(substr)
            if not cols:
                continue

            val = 0.0
            for col in cols:
                v = row.get(col)
                try:
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        continue
                    v_float = float(v)
                    if v_float > val:
                        val = v_float
                except Exception:
                    continue

            if val > max_val:
                return True

        return False

    # ------------------------------------------------------------------
    # DRUG-SPECIFIC PREFERENCE SCORE (for ranking)
    # ------------------------------------------------------------------
    def _preference_score(self, drug_list, food_index: int) -> float:
        """
        Extra scoring component based on positive recommendations
        (e.g., 'drink plenty of fluids' → prefer high-water foods).
        This does NOT affect safety, only ranking.
        """
        if isinstance(drug_list, str):
            drug_list = [drug_list]

        drugs_lower = [d.lower().strip() for d in drug_list]
        score = 0.0

        # Peginterferon alfa-2a: prefer high-water foods
        if any(d == "peginterferon alfa-2a" for d in drugs_lower):
            if self.water_values is not None and self.max_water > 0:
                water_val = self.water_values[food_index]
                normalized = water_val / self.max_water  # 0..1
                score += 0.7 * normalized  # tweak weight as needed

        # Add more drug-specific preferences here later if needed.

        return score

    # ------------------------------------------------------------------
    # PUBLIC API – SAFE FOODS
    # ------------------------------------------------------------------
    def recommend_safe_foods(
        self,
        drug_list,
        gender: str | None = None,
        age: int | None = None,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        bad_terms = self._bad_terms_from_drugs(drug_list)
        nutrient_limits = self._combined_drug_limits(drug_list)

        results: List[Dict[str, Any]] = []

        for i, name in enumerate(self.food_names):
            name_l = name.lower()

            # 1) Exclude direct name conflicts (garlic, ginger, etc.)
            if any(t in name_l for t in bad_terms):
                continue

            # 2) Exclude foods that violate drug-specific nutrient limits
            if self._violates_nutrient_limits(i, nutrient_limits):
                continue

            prob_safe = self._predict_proba_safe_by_index(i)
            ml_safe = self._predict_food_safety_by_index(i)
            pref = self._preference_score(drug_list, i)

            results.append(
                {
                    "food_key": self.food_keys[i],
                    "food_name": name,
                    "ml_safe": ml_safe,
                    "prob_safe": prob_safe,
                    "preference_score": pref,
                    "final_score": prob_safe + pref,
                }
            )

        # Sort by combined (safety + preference), then by probability as tie-breaker
        results.sort(
            key=lambda x: (x["final_score"], x["prob_safe"]),
            reverse=True,
        )
        return results[:top_k]

    # ------------------------------------------------------------------
    # PUBLIC API – DANGEROUS FOODS
    # ------------------------------------------------------------------
    def list_dangerous_foods(
        self,
        drug_list,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        bad_terms = self._bad_terms_from_drugs(drug_list)
        nutrient_limits = self._combined_drug_limits(drug_list)

        results: List[Dict[str, Any]] = []

        for i, name in enumerate(self.food_names):
            name_l = name.lower()

            name_conflict = any(t in name_l for t in bad_terms)
            nutrient_conflict = self._violates_nutrient_limits(i, nutrient_limits)

            if not (name_conflict or nutrient_conflict):
                continue

            prob_safe = self._predict_proba_safe_by_index(i)
            ml_safe = self._predict_food_safety_by_index(i)

            results.append(
                {
                    "food_key": self.food_keys[i],
                    "food_name": name,
                    "ml_safe": ml_safe,
                    "prob_safe": prob_safe,
                    "name_conflict": name_conflict,
                    "nutrient_conflict": nutrient_conflict,
                }
            )

        # Most dangerous first → lowest prob_safe
        results.sort(key=lambda x: x["prob_safe"])
        return results[:top_k]


# ----------------------------------------------------------------------
# MANUAL TEST (run: python -m src.services.recommendation_service)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rec = RecommendationService()

    test_drugs_single = ["Peginterferon alfa-2a"]
    test_drugs_multi = ["Bivalirudin", "Warfarin"]

    def print_safe_and_dangerous(drugs: List[str], top_k: int = 10):
        print("\n" + "=" * 80)
        print(f"Testing recommendations for drugs: {drugs}")
        print("=" * 80)

        safe_foods = rec.recommend_safe_foods(drugs, top_k=top_k)
        print(f"\n✔ SAFE foods (top {len(safe_foods)}):")
        for i, f in enumerate(safe_foods, start=1):
            p = f["prob_safe"]
            p_str = f"{p:.2f}" if isinstance(p, (int, float)) else "N/A"
            fs = f["final_score"]
            print(
                f"{i:2d}. {f['food_name']}  |  P_safe={p_str}  "
                f"|  ML_safe={f['ml_safe']}  |  final_score={fs:.2f}"
            )

        dangerous_foods = rec.list_dangerous_foods(drugs, top_k=top_k)
        print(f"\n❌ DANGEROUS foods (top {len(dangerous_foods)}):")
        for i, f in enumerate(dangerous_foods, start=1):
            p = f["prob_safe"]
            p_str = f"{p:.2f}" if isinstance(p, (int, float)) else "N/A"
            print(
                f"{i:2d}. {f['food_name']}  |  P_safe={p_str}  "
                f"|  ML_safe={f['ml_safe']}  "
                f"|  name_conflict={f['name_conflict']}  "
                f"nutrient_conflict={f['nutrient_conflict']}"
            )

    print_safe_and_dangerous(test_drugs_single, top_k=10)
    print_safe_and_dangerous(test_drugs_multi, top_k=10)
