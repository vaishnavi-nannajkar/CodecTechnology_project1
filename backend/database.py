"""
Database Manager
SQLite-based storage for past search results.
"""

import os
import json
import sqlite3
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sentiment.db")


class DatabaseManager:
    """Manages SQLite storage for sentiment analysis results."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Create tables if they don't exist."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS searches (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        keyword TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        tweet_count INTEGER,
                        positive_count INTEGER,
                        negative_count INTEGER,
                        neutral_count INTEGER,
                        reputation_score REAL,
                        results_json TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_keyword ON searches(keyword);
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"DB init error: {e}")

    def save_results(self, keyword: str, df: pd.DataFrame):
        """Save analysis results to database."""
        try:
            pos = int((df["sentiment"] == "Positive").sum())
            neg = int((df["sentiment"] == "Negative").sum())
            neu = int((df["sentiment"] == "Neutral").sum())
            total = len(df)
            rep_score = (pos - neg) / total if total else 0

            # Store a trimmed version (avoid huge JSON blobs)
            sample = df[["tweet", "sentiment", "confidence"]].head(50)
            results_json = sample.to_json(orient="records")

            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO searches
                        (keyword, timestamp, tweet_count, positive_count,
                         negative_count, neutral_count, reputation_score, results_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    keyword,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    total, pos, neg, neu, rep_score, results_json,
                ))
                conn.commit()
            logger.info(f"Saved results for '{keyword}' to database.")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def get_past_searches(self, limit: int = 20) -> List[Dict]:
        """Retrieve recent past searches."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT keyword, timestamp, tweet_count, reputation_score
                    FROM searches
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                return [
                    {
                        "keyword": r[0],
                        "timestamp": r[1],
                        "count": r[2],
                        "reputation": round(r[3] * 100, 1) if r[3] else 0,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed to retrieve past searches: {e}")
            return []

    def get_search_by_keyword(self, keyword: str) -> Optional[pd.DataFrame]:
        """Get the last saved results for a specific keyword."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT results_json FROM searches
                    WHERE keyword = ?
                    ORDER BY id DESC LIMIT 1
                """, (keyword,))
                row = cursor.fetchone()
                if row:
                    return pd.read_json(row[0])
        except Exception as e:
            logger.error(f"Failed to get search: {e}")
        return None

    def delete_old_records(self, keep_last: int = 100):
        """Remove old records, keeping only the most recent N."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    DELETE FROM searches
                    WHERE id NOT IN (
                        SELECT id FROM searches ORDER BY id DESC LIMIT ?
                    )
                """, (keep_last,))
                conn.commit()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")