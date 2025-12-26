# src/services/recommendation_service.py

from __future__ import annotations
from typing import List, Dict, Any, Set, Optional, Tuple
import re
import numpy as np
import pandas as pd

from src.services.interaction_service import InteractionService
from src.services.nutrient_service import NutrientService


# ----------------------------
# Keyword groups (interpretable rules)
# ----------------------------
RULE_KEYWORDS = {
    "grapefruit": ["grapefruit", "seville orange", "pomelo"],
    "alcohol": ["alcohol", "wine", "beer", "ethanol"],
    "caffeine": ["caffeine", "coffee", "tea", "cola", "energy drink"],
    "dairy_calcium": ["milk", "dairy", "cheese", "yogurt", "calcium"],
    "iron": ["iron"],
    "vitamin_k": ["vitamin k"],
    "tyramine": ["tyramine", "aged cheese", "fermented", "cured meat"],
    "st_johns_wort": ["st. john", "st john", "st john's wort"],
    "ginkgo": ["ginkgo"],
    "ginseng": ["ginseng"],
    "garlic": ["garlic"],
    "ginger": ["ginger"],
}

# Which keyword-groups we treat as "avoid foods with this keyword in name"
# (Simple baseline. You can expand using nutrient tags later.)
KEYWORD_BLOCKLIST_GROUPS = {
    "grapefruit", "alcohol", "caffeine", "dairy_calcium",
    "st_johns_wort", "ginkgo", "ginseng", "garlic", "ginger", "tyramine"
}

# Nutrient-based limits (you already have this idea)
DRUG_NUTRIENT_LIMITS: Dict[str, List[Dict[str, Any]]] = {
    "warfarin": [{"nutrient_substr": "vitamin k", "max_per_100g": 30.0}],
    # Add more later:
    # "spironolactone": [{"nutrient_substr": "potassium (k)", "max_per_100g": 400.0}],
}


class RuleBasedRecommendationService:
    """
    TRUE DSS:
    - Uses DrugBank interaction texts to derive per-drug avoid keyword groups
    - Uses FSANZ nutrient table for nutrient threshold rules
    - Produces COMPLETE safe + unsafe sets with explanations
    """

    def __init__(self):
        self.interaction_service = InteractionService()
        self.nutrient_service = NutrientService()

        self.df = self.nutrient_service.df.copy()

        # Identify foods
        self.food_keys = self.df["Public Food Key"].astype(str).tolist()
        self.food_names = self.df["Food Name"].astype(str).tolist()

        # Build nutrient column index for fast substring lookup
        self._build_nutrient_column_index()

        # Optional: find water column for ranking (not safety)
        self._find_water_column()

        print(f"✔ RuleBasedRecommendationService ready — {len(self.food_keys)} foods")

    # ----------------------------
    # Nutrient column matching
    # ----------------------------
    def _build_nutrient_column_index(self):
        self.nutrient_cols_lower = {c.lower(): c for c in self.df.columns}
        self.substr_to_cols: Dict[str, List[str]] = {}

        substrs: Set[str] = set()
        for rules in DRUG_NUTRIENT_LIMITS.values():
            for r in rules:
                substrs.add(r["nutrient_substr"].lower())

        for substr in substrs:
            cols = [c for c in self.df.columns if substr in c.lower()]
            self.substr_to_cols[substr] = cols

    def _find_water_column(self):
        self.water_col = None
        for c in self.df.columns:
            if "moisture (water" in c.lower():
                self.water_col = c
                break

        if self.water_col:
            s = pd.to_numeric(self.df[self.water_col], errors="coerce").fillna(0.0)
            self.water_values = s.to_numpy(dtype=float)
            self.max_water = float(np.max(self.water_values)) if len(self.water_values) else 0.0
        else:
            self.water_values = None
            self.max_water = 0.0

    # ----------------------------
    # DrugBank → derive avoid groups
    # ----------------------------
    def _derive_avoid_groups_from_drugbank(self, drug_list: List[str]) -> Set[str]:
        avoid_groups: Set[str] = set()

        for drug in drug_list:
            texts = self.interaction_service.get_drug_interactions(drug)
            merged = " ".join([t.lower() for t in texts])

            for group, kws in RULE_KEYWORDS.items():
                if any(kw in merged for kw in kws):
                    # only add groups we actually enforce as avoidance
                    if group in KEYWORD_BLOCKLIST_GROUPS:
                        avoid_groups.add(group)

        return avoid_groups

    def _food_name_hits_group(self, food_name_lower: str, group: str) -> bool:
        kws = RULE_KEYWORDS.get(group, [])
        return any(kw in food_name_lower for kw in kws)

    # ----------------------------
    # Nutrient limit evaluation
    # ----------------------------
    def _combined_nutrient_limits(self, drug_list: List[str]) -> List[Dict[str, Any]]:
        limits: Dict[str, float] = {}
        for d in drug_list:
            rules = DRUG_NUTRIENT_LIMITS.get(d.lower().strip())
            if not rules:
                continue
            for r in rules:
                k = r["nutrient_substr"].lower()
                mx = float(r["max_per_100g"])
                limits[k] = min(limits.get(k, mx), mx)

        return [{"nutrient_substr": k, "max_per_100g": v} for k, v in limits.items()]

    def _check_nutrient_limits(self, idx: int, limits: List[Dict[str, Any]]) -> List[str]:
        """Return list of violation reasons (empty means no nutrient violations)."""
        if not limits:
            return []

        row = self.df.iloc[idx]
        reasons = []

        for rule in limits:
            substr = rule["nutrient_substr"].lower()
            max_val = rule["max_per_100g"]

            cols = self.substr_to_cols.get(substr, [])
            if not cols:
                continue

            # pick max across matched columns (safe behavior)
            val = 0.0
            for col in cols:
                v = row.get(col)
                v = pd.to_numeric(v, errors="coerce")
                if pd.isna(v):
                    continue
                val = max(val, float(v))

            if val > max_val:
                reasons.append(f"nutrient_limit:{substr}={val:.2f}> {max_val:.2f}")

        return reasons

    # ----------------------------
    # Ranking (optional, not safety)
    # ----------------------------
    def _rank_score(self, drug_list: List[str], idx: int) -> float:
        score = 0.0
        drugs_lower = [d.lower().strip() for d in drug_list]

        # Example preference: Peginterferon alfa-2a → prefer high-water foods
        if "peginterferon alfa-2a" in drugs_lower and self.water_values is not None and self.max_water > 0:
            score += (self.water_values[idx] / self.max_water) * 0.7

        return score

    # ----------------------------
    # PUBLIC API: full sets
    # ----------------------------
    def get_safe_and_unsafe_foods(
        self,
        drug_list: List[str] | str,
        top_k_safe: Optional[int] = None,
        top_k_unsafe: Optional[int] = None,
    ) -> Dict[str, Any]:
        if isinstance(drug_list, str):
            drug_list = [drug_list]

        avoid_groups = self._derive_avoid_groups_from_drugbank(drug_list)
        nutrient_limits = self._combined_nutrient_limits(drug_list)

        safe = []
        unsafe = []

        for i, name in enumerate(self.food_names):
            name_l = name.lower()
            reasons = []

            # keyword-based conflicts (from DrugBank)
            for g in avoid_groups:
                if self._food_name_hits_group(name_l, g):
                    reasons.append(f"keyword_conflict:{g}")

            # nutrient-based limits
            reasons.extend(self._check_nutrient_limits(i, nutrient_limits))

            if reasons:
                unsafe.append({
                    "food_key": self.food_keys[i],
                    "food_name": name,
                    "reasons": reasons
                })
            else:
                safe.append({
                    "food_key": self.food_keys[i],
                    "food_name": name,
                    "rank_score": self._rank_score(drug_list, i)
                })

        # sort safe by rank_score (descending), and unsafe by number of reasons (descending)
        safe.sort(key=lambda x: x["rank_score"], reverse=True)
        unsafe.sort(key=lambda x: len(x["reasons"]), reverse=True)

        if top_k_safe is not None:
            safe = safe[:top_k_safe]
        if top_k_unsafe is not None:
            unsafe = unsafe[:top_k_unsafe]

        return {
            "drugs": drug_list,
            "avoid_groups": sorted(list(avoid_groups)),
            "safe_foods": safe,
            "unsafe_foods": unsafe,
            "counts": {"safe": len(safe), "unsafe": len(unsafe)}
        }


# Manual test
if __name__ == "__main__":
    rec = RuleBasedRecommendationService()
    out = rec.get_safe_and_unsafe_foods(["Peginterferon alfa-2a"], top_k_safe=10, top_k_unsafe=10)

    print("\nAvoid groups:", out["avoid_groups"])
    print("\nSAFE sample:")
    for x in out["safe_foods"]:
        print(" -", x["food_name"], "| score", f"{x['rank_score']:.2f}")

    print("\nUNSAFE sample:")
    for x in out["unsafe_foods"]:
        print(" -", x["food_name"], "|", x["reasons"])
