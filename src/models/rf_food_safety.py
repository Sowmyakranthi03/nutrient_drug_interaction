from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.services.nutrient_service import NutrientService
from src.services.interaction_service import InteractionService

MODEL_PATH = "models/rf_food_safety.pkl"
SCALER_PATH = "models/rf_scaler.pkl"
FEATURE_ORDER_PATH = "models/rf_feature_order.json"

os.makedirs("models", exist_ok=True)

# Reuse your keyword logic (keep consistent with RecommendationService)
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

@dataclass
class TrainResult:
    roc_auc: float | None
    report: str

class RFFoodSafetyTrainer:
    """
    Weak-supervision training:
    - Create labels using your rules (unsafe=1 if any conflict, else 0)
    - Train RandomForest on nutrient columns
    """
    def __init__(self):
        self.nutrient_service = NutrientService()
        self.interaction_service = InteractionService()
        self.df = self.nutrient_service.df.copy()

        self.id_cols = {"Public Food Key", "Food Name", "Classification"}

    def _extract_avoid_groups_from_texts(self, texts: List[str]) -> set[str]:
        groups = set()
        for t in texts:
            tl = t.lower()
            for kw in FOOD_KEYWORDS:
                if kw in tl:
                    groups.add(GROUP_MAP.get(kw, kw))
        return groups

    def _keyword_conflicts(self, name: str, avoid_groups: set[str]) -> bool:
        nl = name.lower()
        if "garlic" in avoid_groups and "garlic" in nl:
            return True
        if "ginger" in avoid_groups and "ginger" in nl:
            return True
        if "grapefruit" in avoid_groups and "grapefruit" in nl:
            return True
        if "st_johns_wort" in avoid_groups and ("st john" in nl or "st. john" in nl):
            return True
        if "alcohol" in avoid_groups and any(x in nl for x in ["beer", "wine", "vodka", "whisky", "spirit", "alcoholic"]):
            return True
        if "dairy" in avoid_groups and any(x in nl for x in ["milk", "cheese", "yogurt", "dairy"]):
            return True
        if "caffeine" in avoid_groups and any(x in nl for x in ["coffee", "caffeine"]):
            return True
        return False

    def _violates_limits(self, row: pd.Series, drugs: List[str]) -> bool:
        for d in drugs:
            key = d.strip().lower()
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
                    return True
        return False

    def build_training_data(self, anchor_drugs: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        anchor_drugs: list of drugs used to generate avoid-groups from DrugBank interactions.
        """
        drugs = [d.strip().lower() for d in anchor_drugs if d and d.strip()]
        texts = []
        for d in drugs:
            texts.extend(self.interaction_service.get_drug_interactions(d))

        avoid_groups = self._extract_avoid_groups_from_texts(texts)

        # Select numeric feature columns
        feature_cols = [c for c in self.df.columns if c not in self.id_cols]

        # X from nutrients
        X = self.df[feature_cols].astype(float).values

        # y from rule labels (unsafe=1, safe=0)
        y = []
        for _, row in self.df.iterrows():
            name = str(row["Food Name"])
            unsafe = self._keyword_conflicts(name, avoid_groups) or self._violates_limits(row, drugs)
            y.append(1 if unsafe else 0)

        return X, np.array(y, dtype=int), feature_cols

    def train(self, anchor_drugs: List[str]) -> TrainResult:
        X, y, feature_cols = self.build_training_data(anchor_drugs)

        # scale for stability (not required for RF, but consistent for future models)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            Xs, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
            max_depth=None,
        )
        clf.fit(X_train, y_train)

        # evaluate
        proba = None
        roc = None
        try:
            proba = clf.predict_proba(X_test)[:, 1]
            roc = float(roc_auc_score(y_test, proba))
        except Exception:
            roc = None

        pred = clf.predict(X_test)
        report = classification_report(y_test, pred, digits=4)

        # save artifacts
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        with open(FEATURE_ORDER_PATH, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, indent=2)

        print("✅ RF model saved:", MODEL_PATH)
        print("✅ scaler saved:", SCALER_PATH)
        print("✅ feature order saved:", FEATURE_ORDER_PATH)
        if roc is not None:
            print("ROC-AUC:", roc)
        print(report)

        return TrainResult(roc_auc=roc, report=report)


if __name__ == "__main__":
    # Use some “anchor drugs” to create training labels (weak supervision)
    # Pick drugs that have rich interaction text in your dataset.
    anchors = ["warfarin", "bivalirudin", "lepirudin"]
    trainer = RFFoodSafetyTrainer()
    trainer.train(anchors)
