CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_text TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    model_name TEXT NOT NULL,
    original_word_count INTEGER,
    summary_word_count INTEGER,
    compression_ratio REAL,
    language TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
