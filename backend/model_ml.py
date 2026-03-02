"""
Traditional ML Sentiment Model
TF-IDF + Logistic Regression / SVM with training utilities.
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Graceful sklearn import
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, classification_report
    )
    from sklearn.calibration import CalibratedClassifierCV
    _sklearn_available = True
except ImportError:
    _sklearn_available = False
    logger.warning("scikit-learn not installed. Using rule-based fallback for ML model.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ml_model.pkl")

# ─── Simple Lexicon Fallback ──────────────────────────────────────────────────
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "love",
    "best", "awesome", "happy", "brilliant", "outstanding", "superb", "nice",
    "positive", "win", "success", "perfect", "beautiful", "enjoy", "loved",
    "excited", "impressive", "innovative", "recommend", "helpful", "efficient",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "worst", "poor", "fail",
    "disappointing", "frustrating", "broken", "slow", "useless", "annoying",
    "problem", "issue", "error", "crash", "expensive", "overpriced", "scam",
    "waste", "disappointing", "negative", "ugly", "boring", "mediocre",
}


def rule_based_sentiment(text: str) -> Dict[str, Any]:
    """Lexicon-based fallback when sklearn is unavailable."""
    words = set(text.lower().split())
    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)

    if pos_count > neg_count:
        conf = min(0.5 + pos_count * 0.08, 0.92)
        return {"label": "Positive", "confidence": conf}
    elif neg_count > pos_count:
        conf = min(0.5 + neg_count * 0.08, 0.92)
        return {"label": "Negative", "confidence": conf}
    else:
        return {"label": "Neutral", "confidence": 0.55 + np.random.uniform(0, 0.15)}


# ─── Synthetic Training Data Generator ───────────────────────────────────────
def _generate_training_data():
    """Generate synthetic labeled data for initial model training."""
    positive_tweets = [
        "This product is absolutely amazing and I love it",
        "Great experience with the customer service",
        "Best purchase I have ever made highly recommend",
        "Fantastic quality and fast delivery thank you",
        "Excellent product works perfectly as described",
        "So happy with this wonderful service",
        "Outstanding performance exceeded all expectations",
        "Really impressed with the quality and innovation",
        "Brilliant solution that actually works great",
        "Loving every feature of this awesome product",
        "Perfect gift idea everyone will enjoy it",
        "Super fast and reliable service excellent",
        "Highly recommend this beautiful product",
        "Incredible value for money best deal ever",
        "Very helpful and efficient team great work",
        "Amazing results after using this product daily",
        "The quality is superb and worth every penny",
        "Wonderful experience from start to finish",
        "Absolutely brilliant cannot recommend enough",
        "Five stars deserved truly outstanding service",
    ] * 5

    negative_tweets = [
        "This is terrible and a complete waste of money",
        "Worst experience ever never buying again",
        "Product broke after one day terrible quality",
        "Customer service is awful and unhelpful",
        "Very disappointed with the slow delivery",
        "Horrible product does not work as advertised",
        "Total scam overpriced and low quality",
        "Frustrating experience would not recommend",
        "Bad quality extremely disappointing purchase",
        "Useless product broken on arrival terrible",
        "Annoying issues crash constantly broken",
        "Expensive junk that does not work properly",
        "Negative experience poor customer support",
        "Failed to meet any of my expectations ugly",
        "Mediocre quality not worth the high price",
        "Extremely slow boring and frustrating to use",
        "Terrible service hateful experience never again",
        "Worst product ever regret buying this junk",
        "Horrible quality broke immediately waste money",
        "Very bad experience cannot recommend at all",
    ] * 5

    neutral_tweets = [
        "The product arrived today and I will try it",
        "Just received my order will update later",
        "Ordered yesterday waiting for it to arrive",
        "The item looks ok nothing special so far",
        "Delivery was on time packaging was normal",
        "The product does what it says nothing more",
        "Average quality as expected for the price",
        "It works fine but nothing extraordinary",
        "Received the package today looks standard",
        "The product is okay meets basic requirements",
        "Neither impressed nor disappointed neutral",
        "It is what it is nothing to complain about",
        "Decent product average experience overall",
        "Standard quality as described in listing",
        "Works as expected no issues no surprises",
        "Typical product for this price range okay",
        "Not bad not great somewhere in the middle",
        "Acceptable quality normal delivery time",
        "Meets expectations nothing more nothing less",
        "The product is fine for everyday basic use",
    ] * 5

    texts = positive_tweets + negative_tweets + neutral_tweets
    labels = (
        ["Positive"] * len(positive_tweets)
        + ["Negative"] * len(negative_tweets)
        + ["Neutral"] * len(neutral_tweets)
    )
    return texts, labels


# ─── ML Model Class ───────────────────────────────────────────────────────────
class MLSentimentModel:
    """
    TF-IDF + Logistic Regression sentiment classifier.
    Falls back to rule-based predictions if sklearn is unavailable.
    """

    def __init__(self, model_type: str = "logistic"):
        self.model_type = model_type
        self.pipeline = None
        self._load_or_train()

    def _build_pipeline(self) -> "Pipeline":
        vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=1,
            analyzer="word",
            strip_accents="unicode",
        )
        if self.model_type == "svm":
            base_clf = LinearSVC(max_iter=2000, C=1.0)
            classifier = CalibratedClassifierCV(base_clf)
        else:
            classifier = LogisticRegression(
                max_iter=1000, C=2.0,
                multi_class="multinomial", solver="lbfgs",
            )
        return Pipeline([("tfidf", vectorizer), ("clf", classifier)])

    def _train(self):
        """Train on synthetic data and save model."""
        logger.info("Training ML model on synthetic data...")
        texts, labels = _generate_training_data()
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(texts, labels)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.pipeline, f)
        logger.info(f"ML model saved to {MODEL_PATH}")

    def _load_or_train(self):
        if not _sklearn_available:
            logger.warning("Using rule-based fallback (sklearn unavailable).")
            return
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.pipeline = pickle.load(f)
                logger.info("ML model loaded from cache.")
                return
            except Exception as e:
                logger.warning(f"Could not load cached model: {e}. Retraining.")
        self._train()

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment for a list of cleaned texts."""
        if not _sklearn_available or self.pipeline is None:
            return [rule_based_sentiment(t) for t in texts]

        results = []
        for text in texts:
            if not text.strip():
                results.append({"label": "Neutral", "confidence": 0.60})
                continue
            try:
                label = self.pipeline.predict([text])[0]
                proba = self.pipeline.predict_proba([text])[0]
                confidence = float(np.max(proba))
                results.append({"label": label, "confidence": confidence})
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                results.append(rule_based_sentiment(text))
        return results

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """Evaluate model on labeled data."""
        if not _sklearn_available or self.pipeline is None:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        preds = [self.pipeline.predict([t])[0] for t in texts]
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall": recall_score(labels, preds, average="weighted", zero_division=0),
            "f1": f1_score(labels, preds, average="weighted", zero_division=0),
            "report": classification_report(labels, preds, zero_division=0),
        }

    def retrain(self, texts: List[str], labels: List[str]):
        """Retrain model on new labeled data."""
        if not _sklearn_available:
            return
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(texts, labels)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.pipeline, f)
        logger.info("ML model retrained and saved.")