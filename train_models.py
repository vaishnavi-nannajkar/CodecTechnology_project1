"""
Model Training Script
Train and evaluate ML and BERT models on labeled datasets.

Usage:
    python train_models.py --model ml         # Train TF-IDF + LR
    python train_models.py --model bert       # Fine-tune BERT
    python train_models.py --model both       # Train both
    python train_models.py --evaluate         # Evaluate loaded models
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")

# ─── Sample Datasets ──────────────────────────────────────────────────────────
SAMPLE_DATA = {
    "texts": [
        # Positive
        "This product is absolutely fantastic, highly recommended!",
        "Best service I have ever received, extremely satisfied!",
        "Amazing quality, exceeded all my expectations.",
        "Love this product, it works perfectly and is very reliable.",
        "Excellent customer support, very helpful and responsive.",
        "Outstanding performance, definitely worth every penny.",
        "Incredible value, best purchase I've made this year.",
        "The team is brilliant and delivered beyond expectations.",
        "So happy with this product, will buy again for sure.",
        "Fantastic user experience, intuitive and beautiful design.",
        # Negative
        "Terrible quality, broke after one day. Waste of money!",
        "Worst experience ever. Customer service was rude and unhelpful.",
        "Product is nothing like described. Very disappointed.",
        "Complete scam, do not buy. Overpriced and broken.",
        "Frustrating bugs everywhere, the app crashes constantly.",
        "Awful, awful, awful. Cannot believe I paid for this garbage.",
        "Horrible service, waited 2 weeks and got the wrong item.",
        "The product is poorly made and fell apart immediately.",
        "Disgusting quality control, defective items shipped.",
        "Worst software I've used. Slow, buggy, and unusable.",
        # Neutral
        "Received the package today. Haven't tried it yet.",
        "It's okay, nothing special. Does what it says.",
        "Average product, meets basic requirements.",
        "Delivered on time. Quality seems standard.",
        "Not sure what to think yet. Will update after more use.",
        "Neither great nor bad. Just a standard product.",
        "Works as expected, nothing more nothing less.",
        "The product arrived in good condition. Testing now.",
        "Decent quality for the price. Average performance.",
        "Ordered it, arrived, using it. No major complaints.",
    ],
    "labels": (
        ["Positive"] * 10
        + ["Negative"] * 10
        + ["Neutral"] * 10
    ),
}


def train_ml_model(data_path: str = None):
    """Train the ML model (TF-IDF + Logistic Regression)."""
    logger.info("=" * 50)
    logger.info("Training ML Model (TF-IDF + Logistic Regression)")
    logger.info("=" * 50)

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        from backend.model_ml import MLSentimentModel
        from backend.preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor()

        # Load data
        if data_path and os.path.exists(data_path):
            import pandas as pd
            df = pd.read_csv(data_path)
            texts = df["text"].tolist()
            labels = df["label"].tolist()
            logger.info(f"Loaded {len(texts)} samples from {data_path}")
        else:
            texts = SAMPLE_DATA["texts"]
            labels = SAMPLE_DATA["labels"]
            logger.info(f"Using {len(texts)} built-in sample texts")

        # Preprocess
        logger.info("Preprocessing texts...")
        clean_texts = [preprocessor.clean(t) for t in texts]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            clean_texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Train
        model = MLSentimentModel()
        model.retrain(X_train, y_train)
        logger.info("Model trained successfully!")

        # Evaluate
        logger.info("\nEvaluating on test set...")
        metrics = model.evaluate(X_test, y_test)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1']:.4f}")

        if "report" in metrics:
            logger.info(f"\nClassification Report:\n{metrics['report']}")

        logger.info("✅ ML model training complete!")
        return metrics

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install scikit-learn")
        return None
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None


def train_bert_model(data_path: str = None, epochs: int = 3):
    """Fine-tune BERT model."""
    logger.info("=" * 50)
    logger.info("Fine-tuning BERT Model")
    logger.info("=" * 50)

    try:
        from backend.model_bert import BERTSentimentModel
        from backend.preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor()

        if data_path and os.path.exists(data_path):
            import pandas as pd
            df = pd.read_csv(data_path)
            texts = df["text"].tolist()
            labels = df["label"].tolist()
        else:
            texts = SAMPLE_DATA["texts"]
            labels = SAMPLE_DATA["labels"]

        logger.info(f"Training with {len(texts)} samples for {epochs} epochs")
        logger.warning("⚠️  BERT fine-tuning requires GPU and significant memory!")
        logger.warning("⚠️  Using pre-trained model for small datasets is recommended.")

        model = BERTSentimentModel()
        model.fine_tune(texts, labels, epochs=epochs)

        logger.info("✅ BERT fine-tuning complete!")

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install transformers torch")
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")


def evaluate_models():
    """Evaluate both models on test data."""
    logger.info("=" * 50)
    logger.info("Evaluating Models")
    logger.info("=" * 50)

    from backend.model_ml import MLSentimentModel
    from backend.model_bert import BERTSentimentModel
    from backend.preprocessing import TextPreprocessor

    preprocessor = TextPreprocessor()
    clean_texts = [preprocessor.clean(t) for t in SAMPLE_DATA["texts"]]
    labels = SAMPLE_DATA["labels"]

    # ML evaluation
    logger.info("\n--- ML Model ---")
    ml = MLSentimentModel()
    ml_preds = ml.predict(clean_texts)
    ml_labels = [p["label"] for p in ml_preds]

    correct = sum(p == l for p, l in zip(ml_labels, labels))
    logger.info(f"Accuracy: {correct / len(labels):.4f} ({correct}/{len(labels)})")

    # BERT evaluation
    logger.info("\n--- BERT Model ---")
    bert = BERTSentimentModel()
    bert_preds = bert.predict(clean_texts)
    bert_labels = [p["label"] for p in bert_preds]

    correct_bert = sum(p == l for p, l in zip(bert_labels, labels))
    logger.info(f"Accuracy: {correct_bert / len(labels):.4f} ({correct_bert}/{len(labels)})")


def main():
    parser = argparse.ArgumentParser(description="Train AI Sentiment Models")
    parser.add_argument("--model", choices=["ml", "bert", "both"], default="ml",
                        help="Which model to train")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to CSV file with 'text' and 'label' columns")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs for BERT fine-tuning")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate loaded models instead of training")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_models()
        return

    if args.model in ("ml", "both"):
        train_ml_model(args.data)

    if args.model in ("bert", "both"):
        train_bert_model(args.data, epochs=args.epochs)


if __name__ == "__main__":
    main()