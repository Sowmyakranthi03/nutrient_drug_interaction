from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Set, Tuple

import joblib
import numpy as np
from rapidfuzz import fuzz

from src.services.nutrient_service import NutrientService
from src.services.interaction_service import InteractionService
from src.database.feedback_store import FeedbackStore

# ---------------------------
# RULE KEYWORDS (same as before)
# ---------------------------
FOOD_KEYWORDS = [
    "garlic","ginger","ginseng","ginkgo","ginkgo biloba","chamomile","echinacea",
    "bilberry","danshen","st. john","st john","st johns","st john's",
    "grapefruit","alcohol","wine","beer",
    "milk","dairy","cheese","yogurt",
    "coffee","tea","caffeine",
    "vitamin k","vitamin c",
    "high-fat","high fat",
]

GROUP_MAP = {
    "garlic": "garlic",
    "ginger": "ginger",
    "ginkgo": "ginkgo",
    "ginkgo biloba": "ginkgo",
    "ginseng": "ginseng",
    "chamomile": "chamomile",
    "echinacea": "echinacea",
    "bilberry": "bilberry",
    "danshen": "danshen",
    "st. john": "st_johns_wort",
    "st john": "st_johns_wort",
    "st johns": "st_johns_wort",
    "st john's": "st_johns_wort",
    "grapefruit": "grapefruit",
    "alcohol": "alcohol",
    "wine": "alcohol",
    "beer": "alcohol",
    "milk": "dairy",
    "dairy": "dairy",
    "cheese": "dairy",
    "yogurt": "dairy",
    "coffee": "caffeine",
    "tea": "caffeine",
    "caffeine": "caffeine",
    "vitamin k": "vitamin_k",
}

DRUG_NUTRIENT_LIMITS = {
    "warfarin": [{"substr": "vitamin k", "max": 30.0}],
}

# ---------------------------
# RF ARTIFACT PATHS
# ---------------------------
RF_MODEL_PATH = "models/rf_food_safety.pkl"
RF_SCALER_PATH = "models/rf_scaler.pkl"
RF_FEATURE_ORDER_PATH = "models/rf_feature_order.json"


class RecommendationService:
    """
    Hybrid DSS:
    - Rule-based safety screening (hard constraints)
    - ML (Random Forest) gives unsafe probability to improve ranking
    - FeedbackStore adds personalization bias
    """
    def __init__(self):
        self.nutrient_service = NutrientService()
        self.interaction_service = InteractionService()
        self.feedback_store = FeedbackStore()
        self.df = self.nutrient_service.df

        # Load ML model if available
        self.rf_model = None
        self.rf_scaler = None
        self.rf_features = None
        self._load_rf_if_available()

        # quick indexes
        self.food_keys = self.df["Public Food Key"].astype(str).tolist()
        self.food_names = self.df["Food Name"].astype(str).tolist()
        self.classifications = self.df["Classification"].astype(str).tolist()

        # water column for mild preference (optional)
        self.water_col = None
        for c in self.df.columns:
            if "moisture (water" in c.lower():
                self.water_col = c
                break

        # numeric feature columns available
        self.id_cols = {"Public Food Key", "Food Name", "Classification"}
        self.numeric_cols = [c for c in self.df.columns if c not in self.id_cols]

    def _load_rf_if_available(self):
        if os.path.exists(RF_MODEL_PATH) and os.path.exists(RF_SCALER_PATH) and os.path.exists(RF_FEATURE_ORDER_PATH):
            try:
                self.rf_model = joblib.load(RF_MODEL_PATH)
                self.rf_scaler = joblib.load(RF_SCALER_PATH)
                with open(RF_FEATURE_ORDER_PATH, "r", encoding="utf-8") as f:
                    self.rf_features = json.load(f)
                print("✔ RF model loaded:", RF_MODEL_PATH)
            except Exception as e:
                print("⚠ RF model load failed:", e)
                self.rf_model = None
                self.rf_scaler = None
                self.rf_features = None
        else:
            print("ℹ RF model not found. Using rule-based ranking only.")

    def _match_drug(self, user_drug: str) -> str | None:
        if not user_drug:
            return None
        u = user_drug.strip().lower()
        if u in self.interaction_service.drug_map:
            return u

        best = None
        best_score = 0
        for d in self.interaction_service.drug_map.keys():
            s = fuzz.partial_ratio(u, d)
            if s > best_score:
                best_score = s
                best = d
        return best if best_score >= 80 else None

    def _extract_avoid_groups(self, drugs: List[str]) -> Tuple[Set[str], List[str]]:
        groups: Set[str] = set()
        raw_hits: List[str] = []

        for drug in drugs:
            key = self._match_drug(drug)
            if not key:
                continue
            texts = self.interaction_service.get_drug_interactions(key)
            for t in texts:
                tl = t.lower()
                for kw in FOOD_KEYWORDS:
                    if kw in tl:
                        raw_hits.append(kw)
                        groups.add(GROUP_MAP.get(kw, kw))
        return groups, raw_hits

    def _violates_limits(self, food_idx: int, drugs: List[str]) -> List[str]:
        reasons = []
        row = self.df.iloc[food_idx]
        for d in drugs:
            key = self._match_drug(d)
            if not key:
                continue
            rules = DRUG_NUTRIENT_LIMITS.get(key)
            if not rules:
                continue
            for r in rules:
                substr = r["substr"].lower()
                maxv = float(r["max"])
                cols = [c for c in self.df.columns if substr in c.lower()]
                if not cols:
                    continue
                val = 0.0
                for c in cols:
                    try:
                        val = max(val, float(row.get(c, 0.0)))
                    except Exception:
                        pass
                if val > maxv:
                    reasons.append(f"nutrient_limit:{substr}>{maxv}")
        return reasons

    def _keyword_conflicts(self, name: str, avoid_groups: Set[str]) -> List[str]:
        nl = name.lower()
        reasons = []
        if "garlic" in avoid_groups and "garlic" in nl:
            reasons.append("keyword_conflict:garlic")
        if "ginger" in avoid_groups and "ginger" in nl:
            reasons.append("keyword_conflict:ginger")
        if "grapefruit" in avoid_groups and "grapefruit" in nl:
            reasons.append("keyword_conflict:grapefruit")
        if "st_johns_wort" in avoid_groups and ("st john" in nl or "st. john" in nl):
            reasons.append("keyword_conflict:st_johns_wort")
        if "alcohol" in avoid_groups and any(x in nl for x in ["beer", "wine", "vodka", "whisky", "spirit", "alcoholic"]):
            reasons.append("keyword_conflict:alcohol")
        if "dairy" in avoid_groups and any(x in nl for x in ["milk", "cheese", "yogurt", "dairy"]):
            reasons.append("keyword_conflict:dairy")
        if "caffeine" in avoid_groups and any(x in nl for x in ["coffee", "caffeine"]):
            reasons.append("keyword_conflict:caffeine")
        return reasons

    def _ml_unsafe_probability(self, idx: int) -> float:
        """
        Returns probability that a food is UNSAFE from RF model.
        If model missing, returns neutral 0.5.
        """
        if not self.rf_model or not self.rf_scaler or not self.rf_features:
            return 0.5

        row = self.df.iloc[idx]
        # Ensure we feed exactly the same feature order used in training
        x = []
        for c in self.rf_features:
            try:
                x.append(float(row.get(c, 0.0)))
            except Exception:
                x.append(0.0)

        X = np.array([x], dtype=float)
        Xs = self.rf_scaler.transform(X)

        try:
            proba_unsafe = float(self.rf_model.predict_proba(Xs)[0][1])
            return max(0.0, min(1.0, proba_unsafe))
        except Exception:
            return 0.5

    def _base_score(self, idx: int) -> float:
        score = 0.55
        if self.water_col:
            w = float(self.df.iloc[idx].get(self.water_col, 0.0))
            score += min(0.10, w / 1200.0)  # smaller boost than earlier
        return score

    def recommend(self, drugs: List[str], top_k: int = 50) -> Dict[str, Any]:
        drugs = [d.strip() for d in drugs if d and d.strip()]
        avoid_groups, _ = self._extract_avoid_groups(drugs)
        bias_map = self.feedback_store.get_bias_map(drugs)

        safe = []
        unsafe = []

        for i, name in enumerate(self.food_names):
            food_key = self.food_keys[i]
            reasons = []
            reasons += self._keyword_conflicts(name, avoid_groups)
            reasons += self._violates_limits(i, drugs)

            rule_unsafe = len(reasons) > 0

            # ML probability of UNSAFE
            p_unsafe = self._ml_unsafe_probability(i)

            # score strategy:
            # - base score
            # - penalize by ML unsafe probability
            # - apply feedback bias
            # - heavy penalty if rule unsafe (hard constraint)
            score = self._base_score(i)

            # ML penalty: the more unsafe probability, the lower the score
            score -= 0.40 * p_unsafe

            # feedback bias
            score += bias_map.get(str(food_key), 0.0)

            if rule_unsafe:
                score -= 0.40  # hard constraint penalty

            item = {
                "food_key": str(food_key),
                "food_name": name,
                "classification": self.classifications[i],
                "score": float(max(0.0, min(1.0, score))),
                "ml_unsafe_prob": float(p_unsafe),
                "reasons": reasons
            }

            if rule_unsafe:
                unsafe.append(item)
            else:
                safe.append(item)

        safe.sort(key=lambda x: x["score"], reverse=True)
        unsafe.sort(key=lambda x: x["score"])  # worst first

        def diversify(items, k):
            out = []
            seen = {}
            for it in items:
                cls = it["classification"]
                cnt = seen.get(cls, 0)
                if cnt >= 10:
                    continue
                out.append(it)
                seen[cls] = cnt + 1
                if len(out) >= k:
                    break
            return out

        safe = diversify(safe, top_k)
        unsafe = diversify(unsafe, top_k)

        return {
            "drugs": drugs,
            "avoid_groups": sorted(list(avoid_groups)),
            "safe": safe,
            "unsafe": unsafe,
            "counts": {"safe": len(safe), "unsafe": len(unsafe)},
            "ml": {
                "enabled": bool(self.rf_model is not None),
                "model": "RandomForestClassifier",
                "artifact": RF_MODEL_PATH if self.rf_model else None
            }
        }
