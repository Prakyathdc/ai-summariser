"""
Language Detection Module
==========================

📚 WHAT THIS MODULE DOES:
    Detects the language of input text (e.g., "English", "French", "Hindi").
    This is useful for:
    1. Warning users if their text isn't in English (our models are English-optimized)
    2. Selecting the correct stopword list
    3. Displaying language info in the UI

📚 HOW LANGUAGE DETECTION WORKS:
    The langdetect library (port of Google's language-detection) uses:
    1. N-gram profiles: Character sequences like "th", "the", "he" are common in English
    2. Bayesian classification: Compares the text's n-gram profile against
       profiles of 55 known languages
    3. Confidence score: Higher = more certain about the detection

    Example: "Bonjour le monde" has n-grams like "bo", "on", "nj" → French profile match

📚 LIBRARY:
    langdetect: Python port of Google's language-detection library.
    - detect(text) → "en", "fr", "hi", etc.
    - detect_langs(text) → [en:0.85, fr:0.10, ...] (with probabilities)
"""

from dataclasses import dataclass
import logging
from typing import Optional

from langdetect import detect, detect_langs, LangDetectException

logger = logging.getLogger(__name__)


# Mapping of ISO 639-1 codes to full language names
LANGUAGE_NAMES: dict[str, str] = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian",
    "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
    "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "sq": "Albanian", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "vi": "Vietnamese", "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)",
}


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    language_code: str          # ISO 639-1 code (e.g., "en")
    language_name: str          # Full name (e.g., "English")
    confidence: float           # 0.0 to 1.0
    is_english: bool            # Convenience flag
    all_detected: list[dict]    # All detected languages with probabilities


def detect_language(text: str) -> LanguageDetectionResult:
    """
    Detect the language of the given text.

    Args:
        text: Input text (at least 20 characters recommended for accuracy).

    Returns:
        LanguageDetectionResult with language info and confidence.

    Example:
        >>> result = detect_language("The quick brown fox jumps over the lazy dog.")
        >>> result.language_name
        'English'
        >>> result.confidence > 0.9
        True
    """
    if not text or len(text.strip()) < 10:
        logger.warning("Text too short for reliable language detection")
        return LanguageDetectionResult(
            language_code="en",
            language_name="English (assumed — text too short)",
            confidence=0.0,
            is_english=True,
            all_detected=[],
        )

    try:
        # detect_langs returns a list of Language objects with .lang and .prob
        detected_languages = detect_langs(text[:5000])  # Use first 5000 chars

        # Primary language
        primary = detected_languages[0]
        lang_code = primary.lang
        confidence = primary.prob

        lang_name = LANGUAGE_NAMES.get(lang_code, f"Unknown ({lang_code})")

        # All detected languages for display
        all_detected = [
            {
                "code": lang.lang,
                "name": LANGUAGE_NAMES.get(lang.lang, f"Unknown ({lang.lang})"),
                "probability": round(lang.prob, 3),
            }
            for lang in detected_languages
        ]

        logger.info(
            "Language detected: %s (%s) with %.1f%% confidence",
            lang_name, lang_code, confidence * 100,
        )

        return LanguageDetectionResult(
            language_code=lang_code,
            language_name=lang_name,
            confidence=round(confidence, 3),
            is_english=(lang_code == "en"),
            all_detected=all_detected,
        )

    except LangDetectException as e:
        logger.warning("Language detection failed: %s", e)
        return LanguageDetectionResult(
            language_code="unknown",
            language_name="Unknown",
            confidence=0.0,
            is_english=False,
            all_detected=[],
        )
