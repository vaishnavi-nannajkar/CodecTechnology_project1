"""
FastAPI REST Backend
Optional: use this alongside the Streamlit frontend for external API access.

Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.preprocessing import TextPreprocessor
from backend.model_ml import MLSentimentModel
from backend.model_bert import BERTSentimentModel
from backend.twitter_fetch import TwitterFetcher
from backend.database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Sentiment Intelligence API",
    description="Real-time Twitter sentiment analysis powered by ML and BERT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton instances
preprocessor = TextPreprocessor()
ml_model = MLSentimentModel()
bert_model = BERTSentimentModel()
db = DatabaseManager()


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=100, example="Tesla")
    count: int = Field(default=100, ge=10, le=200)
    model: str = Field(default="ml", pattern="^(ml|bert|both)$")
    use_mock: bool = Field(default=True)


class SentimentResult(BaseModel):
    tweet: str
    clean_tweet: str
    sentiment: str
    confidence: float
    ml_sentiment: Optional[str] = None
    bert_sentiment: Optional[str] = None


class AnalysisResponse(BaseModel):
    keyword: str
    total_tweets: int
    positive: int
    negative: int
    neutral: int
    reputation_score: float
    results: List[SentimentResult]
    model_used: str
    timestamp: str


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=500)
    model: str = Field(default="ml", pattern="^(ml|bert)$")


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ─── Analyze ─────────────────────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Full pipeline: fetch tweets → preprocess → predict → return results."""
    try:
        fetcher = TwitterFetcher(use_mock=request.use_mock)
        raw_tweets = fetcher.fetch(request.keyword, request.count)

        clean_tweets = [preprocessor.clean(t) for t in raw_tweets]

        results = []

        if request.model in ("ml", "both"):
            ml_preds = ml_model.predict(clean_tweets)
        if request.model in ("bert", "both"):
            bert_preds = bert_model.predict(clean_tweets)

        for i, (raw, clean) in enumerate(zip(raw_tweets, clean_tweets)):
            item = {"tweet": raw, "clean_tweet": clean}

            if request.model == "ml":
                p = ml_preds[i]
                item.update(sentiment=p["label"], confidence=p["confidence"],
                            ml_sentiment=p["label"])
            elif request.model == "bert":
                p = bert_preds[i]
                item.update(sentiment=p["label"], confidence=p["confidence"],
                            bert_sentiment=p["label"])
            else:  # both — ensemble
                mp = ml_preds[i]
                bp = bert_preds[i]
                if mp["label"] == bp["label"]:
                    label = mp["label"]
                    conf = (mp["confidence"] + bp["confidence"]) / 2
                else:
                    label = bp["label"]  # BERT wins
                    conf = bp["confidence"]
                item.update(
                    sentiment=label, confidence=conf,
                    ml_sentiment=mp["label"], bert_sentiment=bp["label"],
                )

            results.append(SentimentResult(**item))

        pos = sum(1 for r in results if r.sentiment == "Positive")
        neg = sum(1 for r in results if r.sentiment == "Negative")
        neu = sum(1 for r in results if r.sentiment == "Neutral")
        total = len(results)
        rep_score = (pos - neg) / total if total else 0

        # Save to DB in background
        import pandas as pd
        df = pd.DataFrame([r.dict() for r in results])
        background_tasks.add_task(db.save_results, request.keyword, df)

        return AnalysisResponse(
            keyword=request.keyword,
            total_tweets=total,
            positive=pos,
            negative=neg,
            neutral=neu,
            reputation_score=round(rep_score, 4),
            results=results,
            model_used=request.model,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Predict (single/batch) ───────────────────────────────────────────────────
@app.post("/predict")
async def predict(request: PredictRequest):
    """Predict sentiment for custom text inputs (no tweet fetching)."""
    try:
        clean = [preprocessor.clean(t) for t in request.texts]
        if request.model == "bert":
            preds = bert_model.predict(clean)
        else:
            preds = ml_model.predict(clean)

        return {
            "results": [
                {
                    "original": orig,
                    "clean": cl,
                    "sentiment": p["label"],
                    "confidence": p["confidence"],
                }
                for orig, cl, p in zip(request.texts, clean, preds)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── History ──────────────────────────────────────────────────────────────────
@app.get("/history")
async def history(limit: int = Query(default=20, le=100)):
    """Get recent search history from database."""
    return {"history": db.get_past_searches(limit=limit)}


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)