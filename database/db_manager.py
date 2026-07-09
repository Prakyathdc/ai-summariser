"""
Database Manager Module
=========================
Handles SQLite database connection and CRUD operations for summarization history.
"""

import sqlite3
import logging
from typing import Optional

from config import DATABASE_PATH, DATABASE_DIR

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database operations."""
    
    def __init__(self, db_path: str = str(DATABASE_PATH)):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                schema_path = DATABASE_DIR / "schema.sql"
                with open(schema_path, "r", encoding="utf-8") as f:
                    conn.executescript(f.read())
                logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize database: %s", e)

    def save_summary(self, original_text: str, summary_text: str, model_name: str,
                     original_word_count: int, summary_word_count: int,
                     compression_ratio: float, language: str = "en") -> Optional[int]:
        """Save a new summary to history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''INSERT INTO summaries (
                        original_text, summary_text, model_name, 
                        original_word_count, summary_word_count, 
                        compression_ratio, language
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (original_text, summary_text, model_name, original_word_count,
                     summary_word_count, compression_ratio, language)
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error("Failed to save summary: %s", e)
            return None

    def get_all_summaries(self):
        """Retrieve all summaries from history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM summaries ORDER BY created_at DESC")
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Failed to retrieve summaries: %s", e)
            return []
