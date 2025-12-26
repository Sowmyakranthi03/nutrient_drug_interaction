# src/database/feedback_store.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import json
import time


@dataclass
class FeedbackEvent:
    ts: float
    drugs_key: str
    food_key: str
    vote: int   # +1 or -1


class FeedbackStore:
    """
    Stores user feedback (upvote/downvote) per (drugs_set, food_key).

    - vote: +1 (upvote), -1 (downvote)
    - get_bias_map() returns {food_key: bias_score} used by ranking.
    """

    def __init__(self, path: str = "data/feedback.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"events": []}, indent=2), encoding="utf-8")

    def _normalize_drugs(self, drugs: List[str]) -> str:
        # stable key regardless of order/case/spaces
        clean = sorted({(d or "").strip().lower() for d in drugs if (d or "").strip()})
        return "|".join(clean)

    def _load(self) -> Dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"events": []}

    def _save(self, data: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_feedback(self, drugs: List[str], food_key: str, vote: int) -> None:
        """
        vote must be +1 or -1
        """
        if vote not in (1, -1):
            raise ValueError("vote must be +1 or -1")

        drugs_key = self._normalize_drugs(drugs)
        fk = str(food_key).strip()

        data = self._load()
        events = data.get("events", [])
        events.append(
            {
                "ts": time.time(),
                "drugs_key": drugs_key,
                "food_key": fk,
                "vote": int(vote),
            }
        )
        data["events"] = events
        self._save(data)

    def get_bias_map(self, drugs: List[str]) -> Dict[str, float]:
        """
        Returns bias per food_key for the given drug set.
        A simple scoring:
          bias = 0.06 * (upvotes - downvotes), clipped to [-0.25, +0.25]
        """
        drugs_key = self._normalize_drugs(drugs)
        data = self._load()
        events = data.get("events", [])

        tally: Dict[str, int] = {}
        for e in events:
            if e.get("drugs_key") != drugs_key:
                continue
            fk = str(e.get("food_key", "")).strip()
            v = int(e.get("vote", 0))
            if v not in (1, -1):
                continue
            tally[fk] = tally.get(fk, 0) + v

        bias: Dict[str, float] = {}
        for fk, score in tally.items():
            b = 0.06 * float(score)
            if b > 0.25:
                b = 0.25
            if b < -0.25:
                b = -0.25
            bias[fk] = b

        return bias
