from __future__ import annotations
from typing import List, Dict, Any, Set, Tuple
import numpy as np
from rapidfuzz import fuzz
from src.services.nutrient_service import NutrientService
from src.services.interaction_service import InteractionService
from src.database.feedback_store import FeedbackStore

# keyword groups for extraction
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

# optional nutrient limits demo
DRUG_NUTRIENT_LIMITS = {
    "warfarin": [{"substr": "vitamin k", "max": 30.0}],
}

class RecommendationService:
    def __init__(self):
        self.nutrient_service = NutrientService()
        self.interaction_service = InteractionService()
        self.feedback_store = FeedbackStore()
        self.df = self.nutrient_service.df

        # water column for hydration preference
        self.water_col = None
        for c in self.df.columns:
            if "moisture (water" in c.lower():
                self.water_col = c
                break

        # quick indexes
        self.food_keys = self.df["Public Food Key"].astype(str).tolist()
        self.food_names = self.df["Food Name"].astype(str).tolist()
        self.classifications = self.df["Classification"].astype(str).tolist()

    def _match_drug(self, user_drug: str) -> str | None:
        """fuzzy match typed drug to dataset drug keys"""
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
        # group-to-keywords check
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

    def _base_score(self, idx: int) -> float:
        # make ranking not “only water”
        score = 0.55
        if self.water_col:
            w = float(self.df.iloc[idx].get(self.water_col, 0.0))
            score += min(0.15, w / 1000.0)  # small boost
        # small diversity boost by class
        cls = self.classifications[idx]
        if cls.startswith("29") or cls.startswith("31"):  # beverages/spices etc: don’t dominate
            score -= 0.05
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

            # rule checks
            reasons += self._keyword_conflicts(name, avoid_groups)
            reasons += self._violates_limits(i, drugs)

            is_unsafe = len(reasons) > 0

            score = self._base_score(i)
            # apply feedback bias
            score += bias_map.get(str(food_key), 0.0)
            # unsafe penalty
            if is_unsafe:
                score -= 0.35

            item = {
                "food_key": str(food_key),
                "food_name": name,
                "classification": self.classifications[i],
                "score": float(max(0.0, min(1.0, score))),
                "reasons": reasons
            }

            if is_unsafe:
                unsafe.append(item)
            else:
                safe.append(item)

        # sort
        safe.sort(key=lambda x: x["score"], reverse=True)
        unsafe.sort(key=lambda x: x["score"])  # most unsafe first

        # diversity: take top foods but avoid too many from same classification
        def diversify(items, k):
            out = []
            seen = {}
            for it in items:
                cls = it["classification"]
                cnt = seen.get(cls, 0)
                if cnt >= 10:   # cap per class
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
            "counts": {"safe": len(safe), "unsafe": len(unsafe)}
        }
