from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any

from src.services.recommendation_service import RecommendationService


app = FastAPI()
@app.get("/")
def root():
    return {
        "service": "Nutrient–Drug Interaction API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }
from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    # redirect to swagger docs
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico")
def favicon():
    return {"ok": True}


# React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://nutrient-drug-interaction.vercel.app/",
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
    # show all drug names available in DrugBank json
    return {"drugs": sorted(list(rec.interaction_service.drug_map.keys()))}


@app.post("/recommend")
def recommend(payload: RecommendRequest) -> Dict[str, Any]:
    return rec.recommend(payload.drugs, top_k=payload.top_k)


@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    # You already built FeedbackStore – call its API here
    # I’m assuming these names; if your FeedbackStore uses different names,
    # just replace these two calls.

    if payload.vote == "up":
        rec.feedback_store.add_feedback(payload.drugs, payload.food_key, +1)
    else:
        rec.feedback_store.add_feedback(payload.drugs, payload.food_key, -1)

    return {"status": "ok"}
