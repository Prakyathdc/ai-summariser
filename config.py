"""
AI Text Summarization System — Configuration Module
=====================================================

📚 WHAT THIS FILE DOES:
    This is the "single source of truth" for all settings in the project.
    Instead of scattering magic numbers, file paths, and model names across
    dozens of files, we centralize them here. Any file that needs a setting
    imports it from config.py.

📚 WHY CENTRALIZE CONFIGURATION?
    1. Change a model name in ONE place → entire app updates
    2. No "magic numbers" buried in code
    3. Easy to switch between development and production settings
    4. Interviewers love seeing this — it shows systems thinking

📚 DESIGN PATTERN: Configuration Object
    We use Python dataclasses and a module-level singleton pattern.
    - dataclass: Auto-generates __init__, __repr__, and more from field definitions
    - Paths use pathlib.Path: Cross-platform (Windows/Mac/Linux) path handling
    - Type hints on every field: Makes code self-documenting

📚 LIBRARIES USED:
    - pathlib.Path: Modern, object-oriented filesystem paths (built-in)
    - dataclasses: Reduce boilerplate for data-holding classes (built-in)
    - logging: Python's built-in logging framework (better than print statements)
    - os: Access environment variables (built-in)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
import logging
import os


# ──────────────────────────────────────────────
# 📁 PATH CONFIGURATION
# ──────────────────────────────────────────────
# Path(__file__) gives the path to THIS file (config.py).
# .parent gives its parent directory (AI_Text_Summarizer/).
# .resolve() converts to an absolute path, removing any ".." segments.
# This works on Windows, Mac, and Linux — unlike hardcoded strings.

BASE_DIR: Final[Path] = Path(__file__).parent.resolve()

UPLOAD_DIR: Final[Path] = BASE_DIR / "uploads"
EXPORT_DIR: Final[Path] = BASE_DIR / "exports"
LOG_DIR: Final[Path] = BASE_DIR / "logs"
HISTORY_DIR: Final[Path] = BASE_DIR / "history"
ASSETS_DIR: Final[Path] = BASE_DIR / "assets"
DATABASE_DIR: Final[Path] = BASE_DIR / "database"

# Database file path
DATABASE_PATH: Final[Path] = DATABASE_DIR / "summarizer.db"

# ──────────────────────────────────────────────
# 📁 CREATE DIRECTORIES IF THEY DON'T EXIST
# ──────────────────────────────────────────────
# exist_ok=True means "don't throw an error if the directory already exists".
# parents=True means "create parent directories too if needed".

for directory in [UPLOAD_DIR, EXPORT_DIR, LOG_DIR, HISTORY_DIR, ASSETS_DIR, DATABASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 🤖 MODEL CONFIGURATION
# ──────────────────────────────────────────────
# Each model has different strengths. We define them as a registry
# so the UI can dynamically list available models.

@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for a single summarization model.

    📚 WHY frozen=True?
        Makes instances immutable (can't accidentally modify settings at runtime).
        This is a best practice for configuration objects.

    Attributes:
        name: Hugging Face model identifier (used by AutoTokenizer/AutoModel)
        display_name: Human-readable name for the UI
        description: Brief explanation of the model's strengths
        max_input_length: Maximum tokens the model can process
        default_max_length: Default maximum tokens for generated summary
        default_min_length: Default minimum tokens for generated summary
        default_num_beams: Default beam search width
    """
    name: str
    display_name: str
    description: str
    max_input_length: int
    default_max_length: int = 150
    default_min_length: int = 30
    default_num_beams: int = 4


# Registry of all supported models
# 📚 WHY a tuple (not list)?
#     Tuples are immutable — prevents accidental modification.
AVAILABLE_MODELS: Final[tuple[ModelConfig, ...]] = (
    ModelConfig(
        name="t5-small",
        display_name="T5 Small (Fast, Lightweight)",
        description="Google's Text-to-Text Transfer Transformer. Small but capable. "
                    "Best for development and testing. ~242 MB.",
        max_input_length=512,
        default_max_length=150,
        default_min_length=30,
        default_num_beams=4,
    ),
    ModelConfig(
        name="facebook/bart-large-cnn",
        display_name="BART Large CNN (Best for News)",
        description="Facebook's BART fine-tuned on CNN/DailyMail dataset. "
                    "Excellent for news article summarization. ~1.6 GB.",
        max_input_length=1024,
        default_max_length=142,
        default_min_length=56,
        default_num_beams=4,
    ),
    ModelConfig(
        name="google/pegasus-xsum",
        display_name="PEGASUS XSum (Extreme Summarization)",
        description="Google's PEGASUS fine-tuned on XSum dataset. "
                    "Produces very concise, single-sentence summaries. ~2.2 GB.",
        max_input_length=512,
        default_max_length=64,
        default_min_length=11,
        default_num_beams=8,
    ),
)

# Default model for first-time users (t5-small for fast loading)
DEFAULT_MODEL_NAME: Final[str] = "t5-small"


# ──────────────────────────────────────────────
# 📏 SUMMARY LENGTH PRESETS
# ──────────────────────────────────────────────
# Users choose "Short", "Medium", or "Long" — we map these to token counts.

@dataclass(frozen=True)
class SummaryLengthPreset:
    """Defines token limits for a summary length option."""
    label: str
    max_length: int
    min_length: int


SUMMARY_LENGTHS: Final[dict[str, SummaryLengthPreset]] = {
    "Short": SummaryLengthPreset(label="Short (~50 words)", max_length=75, min_length=20),
    "Medium": SummaryLengthPreset(label="Medium (~150 words)", max_length=200, min_length=50),
    "Long": SummaryLengthPreset(label="Long (~300 words)", max_length=400, min_length=100),
}


# ──────────────────────────────────────────────
# ⚙️ GENERATION PARAMETERS (Defaults)
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class GenerationDefaults:
    """
    Default parameters for text generation.

    📚 PARAMETER EXPLANATIONS:

    num_beams (int):
        Beam search width. At each step, the model keeps this many
        candidate sequences. Higher = better quality, slower.
        - 1 = greedy decoding (fastest, lowest quality)
        - 4 = good balance (our default)
        - 8+ = diminishing returns

    length_penalty (float):
        Controls output length preference.
        - > 1.0: Encourages LONGER outputs
        - < 1.0: Encourages SHORTER outputs
        - = 1.0: No preference

    temperature (float):
        Controls randomness in token selection.
        - 0.0: Always pick the most probable token (deterministic)
        - 1.0: Sample according to model probabilities (default)
        - > 1.0: More random/creative (risky for summarization)

    top_k (int):
        Only consider the top-K most probable tokens at each step.
        - 50: Default, good balance
        - Lower: More focused, less diverse

    top_p (float):
        Nucleus sampling — consider tokens whose cumulative probability
        exceeds this threshold.
        - 0.95: Default, slight filtering of unlikely tokens
        - Lower: More focused

    early_stopping (bool):
        Stop generation when all beam hypotheses have reached the
        end-of-sequence token. Saves computation.

    repetition_penalty (float):
        Penalizes tokens that have already appeared.
        - 1.0: No penalty
        - > 1.0: Reduces repetition (useful for longer summaries)
    """
    num_beams: int = 4
    length_penalty: float = 2.0
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    early_stopping: bool = True
    repetition_penalty: float = 1.0


GENERATION_DEFAULTS: Final[GenerationDefaults] = GenerationDefaults()


# ──────────────────────────────────────────────
# 📊 UI CONFIGURATION
# ──────────────────────────────────────────────

APP_TITLE: Final[str] = "AI Text Summarizer"
APP_ICON: Final[str] = "📝"
APP_DESCRIPTION: Final[str] = (
    "Summarize text, PDFs, DOCX files, web pages, and YouTube videos "
    "using state-of-the-art Transformer models."
)

# Reading speed assumption (words per minute) for reading time estimates
AVERAGE_READING_SPEED_WPM: Final[int] = 200

# Maximum file upload size in MB
MAX_UPLOAD_SIZE_MB: Final[int] = 50


# ──────────────────────────────────────────────
# 📋 LOGGING CONFIGURATION
# ──────────────────────────────────────────────
# 📚 WHY LOGGING instead of print()?
#     - Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL — filter what you see
#     - Timestamps: Know when things happened
#     - File output: Logs persist after the program exits
#     - Production standard: Every professional Python app uses logging

def setup_logging() -> None:
    """
    Configure application-wide logging.

    Creates two handlers:
        1. Console handler — Shows INFO+ messages in terminal
        2. File handler — Writes DEBUG+ messages to logs/app.log

    The format includes timestamp, logger name, level, and message.
    """
    log_file = LOG_DIR / "app.log"

    # Root logger configuration
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            # Console: Show INFO and above
            logging.StreamHandler(),
            # File: Write everything (DEBUG and above)
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)


# ──────────────────────────────────────────────
# 🗃️ SUPPORTED FILE TYPES
# ──────────────────────────────────────────────

SUPPORTED_FILE_EXTENSIONS: Final[tuple[str, ...]] = (".pdf", ".docx", ".txt")


# ──────────────────────────────────────────────
# 🌐 SPACY MODEL
# ──────────────────────────────────────────────

SPACY_MODEL: Final[str] = "en_core_web_sm"
