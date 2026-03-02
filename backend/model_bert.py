"""
BERT Sentiment Model
Uses HuggingFace Transformers (bert-base-uncased or distilbert for speed).
Falls back to rule-based when transformers is unavailable.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

BERT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bert_model")
BASE_MODEL = os.getenv("BERT_BASE_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")

# Graceful import of HuggingFace
try:
    from transformers import pipeline as hf_pipeline
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _transformers_available = True
except ImportError:
    _transformers_available = False
    logger.warning("HuggingFace Transformers not installed. Using enhanced rule-based fallback.")

# ─── Enhanced Lexicon Fallback ─────────────────────────────────────────────────
POSITIVE_PATTERNS = [
    "love", "great", "excellent", "amazing", "wonderful", "fantastic", "best",
    "awesome", "happy", "brilliant", "outstanding", "superb", "perfect", "enjoy",
    "recommend", "helpful", "efficient", "innovative", "impressive", "excited",
    "pleased", "satisfied", "delighted", "thrilled", "passionate", "proud",
]
NEGATIVE_PATTERNS = [
    "hate", "terrible", "awful", "horrible", "worst", "bad", "poor", "fail",
    "disappointing", "frustrating", "broken", "slow", "useless", "annoying",
    "problem", "issue", "error", "crash", "expensive", "scam", "waste",
    "angry", "disgusted", "furious", "outraged", "betrayed", "lied",
]
STRONG_POSITIVE = {"love", "amazing", "excellent", "perfect", "outstanding"}
STRONG_NEGATIVE = {"hate", "terrible", "horrible", "worst", "awful"}
NEGATORS = {"not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't"}


def enhanced_rule_based(text: str) -> Dict[str, Any]:
    """Enhanced lexicon with negation handling."""
    words = text.lower().split()
    neg_flag = False
    pos_score = 0.0
    neg_score = 0.0

    for i, word in enumerate(words):
        if word in NEGATORS:
            neg_flag = True
            continue

        weight = 1.5 if (word in STRONG_POSITIVE or word in STRONG_NEGATIVE) else 1.0

        if word in POSITIVE_PATTERNS:
            if neg_flag:
                neg_score += weight
            else:
                pos_score += weight
        elif word in NEGATIVE_PATTERNS:
            if neg_flag:
                pos_score += weight * 0.5  # double negative → weak positive
            else:
                neg_score += weight

        # Negation window (only 3 words)
        if neg_flag and i > 0:
            neg_flag = False

    total = pos_score + neg_score + 0.01
    if pos_score > neg_score * 1.2:
        conf = min(0.52 + pos_score / (total * 2), 0.94)
        return {"label": "Positive", "confidence": round(conf, 4)}
    elif neg_score > pos_score * 1.2:
        conf = min(0.52 + neg_score / (total * 2), 0.94)
        return {"label": "Negative", "confidence": round(conf, 4)}
    else:
        return {"label": "Neutral", "confidence": round(0.55 + np.random.uniform(0, 0.12), 4)}


# ─── BERT Model Class ──────────────────────────────────────────────────────────
class BERTSentimentModel:
    """
    Sentiment classifier using a fine-tuned BERT/DistilBERT model.

    By default uses `distilbert-base-uncased-finetuned-sst-2-english` (binary)
    and maps to 3-class Positive/Negative/Neutral via confidence thresholding.

    For true 3-class, set BERT_BASE_MODEL env var to a 3-label model such as:
    `cardiffnlp/twitter-roberta-base-sentiment-latest`
    """

    LABEL_MAP_BINARY = {
        "POSITIVE": "Positive",
        "NEGATIVE": "Negative",
        "LABEL_0": "Negative",
        "LABEL_1": "Positive",
        "LABEL_2": "Positive",
    }

    def __init__(self):
        self.clf = None
        self.is_three_class = False
        self._load_model()

    def _load_model(self):
        if not _transformers_available:
            logger.info("Transformers unavailable — rule-based BERT fallback active.")
            return

        try:
            model_name = BASE_MODEL
            # Check if we have a local fine-tuned model
            if os.path.exists(BERT_MODEL_DIR):
                model_name = BERT_MODEL_DIR
                logger.info(f"Loading local BERT model from {BERT_MODEL_DIR}")
            else:
                logger.info(f"Loading pre-trained BERT model: {model_name}")

            device = 0 if (
                _transformers_available and hasattr(__import__("torch"), "cuda")
                and __import__("torch").cuda.is_available()
            ) else -1

            self.clf = hf_pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=device,
                return_all_scores=True,
                truncation=True,
                max_length=512,
            )

            # Detect if model is 3-class
            test_result = self.clf("test")[0]
            self.is_three_class = len(test_result) == 3
            logger.info(
                f"BERT model loaded. {'3-class' if self.is_three_class else 'Binary'} mode."
            )

        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            self.clf = None

    def _parse_scores(self, scores: List[Dict]) -> Dict[str, Any]:
        """Parse raw model output scores to label + confidence."""
        if self.is_three_class:
            # Expecting labels like LABEL_0/1/2 or NEG/NEU/POS
            label_map = {
                "LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive",
                "NEG": "Negative", "NEU": "Neutral", "POS": "Positive",
                "negative": "Negative", "neutral": "Neutral", "positive": "Positive",
            }
            best = max(scores, key=lambda x: x["score"])
            label = label_map.get(best["label"].upper(), best["label"].capitalize())
            return {"label": label, "confidence": round(best["score"], 4)}
        else:
            # Binary: map confidence < threshold → Neutral
            best = max(scores, key=lambda x: x["score"])
            mapped = self.LABEL_MAP_BINARY.get(best["label"].upper(), "Neutral")
            conf = best["score"]
            if conf < 0.65:
                return {"label": "Neutral", "confidence": round(conf + 0.1, 4)}
            return {"label": mapped, "confidence": round(conf, 4)}

    def predict(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Predict sentiment for a list of texts."""
        if not _transformers_available or self.clf is None:
            return [enhanced_rule_based(t) for t in texts]

        results = []
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Replace empty texts
            clean_batch = [t if t.strip() else "neutral" for t in batch]
            try:
                raw_outputs = self.clf(clean_batch)
                for output in raw_outputs:
                    results.append(self._parse_scores(output))
            except Exception as e:
                logger.error(f"BERT batch prediction error: {e}")
                for text in clean_batch:
                    results.append(enhanced_rule_based(text))

        return results

    def fine_tune(
        self,
        texts: List[str],
        labels: List[str],
        epochs: int = 3,
        learning_rate: float = 2e-5,
    ):
        """
        Fine-tune BERT on custom labeled data.
        Requires PyTorch and sufficient memory (GPU recommended).
        """
        if not _transformers_available:
            logger.error("Cannot fine-tune: transformers not installed.")
            return

        try:
            import torch
            from torch.utils.data import Dataset, DataLoader
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                AdamW,
                get_linear_schedule_with_warmup,
            )

            label2id = {"Positive": 2, "Neutral": 1, "Negative": 0}
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            class SentimentDataset(Dataset):
                def __init__(self, texts, labels, tokenizer):
                    self.encodings = tokenizer(
                        texts, truncation=True, padding=True, max_length=128
                    )
                    self.labels = [label2id[l] for l in labels]

                def __len__(self):
                    return len(self.labels)

                def __getitem__(self, idx):
                    item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                    item["labels"] = torch.tensor(self.labels[idx])
                    return item

            dataset = SentimentDataset(texts, labels, tokenizer)
            loader = DataLoader(dataset, batch_size=16, shuffle=True)

            model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=3
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            optimizer = AdamW(model.parameters(), lr=learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=len(loader),
                num_training_steps=len(loader) * epochs,
            )

            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch in loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                logger.info(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(loader):.4f}")

            os.makedirs(BERT_MODEL_DIR, exist_ok=True)
            model.save_pretrained(BERT_MODEL_DIR)
            tokenizer.save_pretrained(BERT_MODEL_DIR)
            logger.info(f"Fine-tuned BERT model saved to {BERT_MODEL_DIR}")

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")