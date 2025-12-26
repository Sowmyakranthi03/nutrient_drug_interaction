from __future__ import annotations

from typing import List, Literal, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.services.recommendation_service import RecommendationService

app = FastAPI(title="Nutrient–Drug Interaction API")

# ✅ Root: redirect to Swagger docs
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# ✅ Avoid noisy 404 logs from bots
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return {"ok": True}

# ✅ CORS (IMPORTANT: no trailing slash)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://nutrient-drug-interaction.vercel.app",
        # If you have a custom domain later, add it here too
        # "https://your-domain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rec = RecommendationService()


class RecommendRequest(BaseModel):
    drugs: List[str]
    top_k: int = 30


class FeedbackRequest(BaseModel):
    drugs: List[str]
    food_key: str
    vote: Literal["up", "down"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/drugs")
def drugs():
    return {"drugs": sorted(list(rec.interaction_service.drug_map.keys()))}


@app.post("/recommend")
def recommend(payload: RecommendRequest) -> Dict[str, Any]:
    return rec.recommend(payload.drugs, top_k=payload.top_k)


@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    # vote: "up" -> +1, "down" -> -1
    delta = +1 if payload.vote == "up" else -1
    rec.feedback_store.add_feedback(payload.drugs, payload.food_key, delta)
    return {"status": "ok"}
