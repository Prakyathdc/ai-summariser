"""
Text Preprocessor Module — Cleaning, Normalization, NLP Processing
====================================================================

📚 WHAT THIS MODULE DOES:
    Takes raw extracted text (from PDFs, DOCX, etc.) and cleans it up.
    Raw text often has:
    - Extra whitespace, newlines, tabs
    - Special characters (bullet points, smart quotes, em-dashes)
    - Unicode artifacts (zero-width spaces, non-breaking spaces)
    - HTML entities (&amp;, &lt;, etc.)

    This module provides a pipeline of cleaning steps that produces
    clean, normalized text ready for summarization.

📚 KEY NLP CONCEPTS TAUGHT:

    TOKENIZATION — Breaking text into tokens (words, subwords, sentences)
    ──────────────────────────────────────────────────────────────────────
    Three levels:
    1. Word tokenization:     "I can't" → ["I", "ca", "n't"] (NLTK) or ["I", "can", "'t"]
    2. Subword tokenization:  "unhappiness" → ["un", "happi", "ness"] (BPE, used by BART/T5)
    3. Sentence tokenization: "Hello. How are you?" → ["Hello.", "How are you?"]

    WHY SUBWORD?
    - Handles unknown words: "COVID-19" → ["CO", "VID", "-", "19"]
    - Finite vocabulary: Instead of millions of words, ~30K-50K subword units
    - This is what BART, T5, PEGASUS use internally

    NORMALIZATION — Making text consistent
    ──────────────────────────────────────
    - Unicode normalization: "café" (NFC) vs "café" (NFD) → same bytes
    - Case normalization: "The" vs "the" (context-dependent)
    - Whitespace normalization: "hello     world" → "hello world"

    STOPWORD REMOVAL — Removing common words
    ──────────────────────────────────────────
    Words like "the", "is", "at", "and" appear in almost every sentence.
    They're useful for grammar but not for meaning.

    ⚠️ IMPORTANT: We do NOT remove stopwords before summarization!
    Transformer models need the full text (including stopwords) to understand
    grammar and context. Stopword removal is only for keyword extraction and
    analysis features.

    LEMMATIZATION vs STEMMING
    ──────────────────────────
    Both reduce words to their base form, but differently:

    Stemming (rule-based, NLTK Porter Stemmer):
        "running" → "run"     ✓
        "studies" → "studi"   ✗ (not a real word!)
        "better"  → "better"  ✗ (doesn't understand irregulars)

    Lemmatization (dictionary + grammar, spaCy):
        "running" → "run"     ✓
        "studies" → "study"   ✓
        "better"  → "good"    ✓ (understands irregular forms!)

    Trade-off: Stemming is faster, lemmatization is more accurate.

📚 LIBRARIES USED:
    - re: Regular expressions for pattern matching (built-in)
    - unicodedata: Unicode character properties (built-in)
    - html: Unescape HTML entities (built-in)
    - nltk: Sentence tokenization, stopwords, stemming
    - spacy: Lemmatization, part-of-speech tagging
"""

from dataclasses import dataclass
from typing import Optional
import html
import logging
import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# NLTK DATA DOWNLOAD
# ══════════════════════════════════════════════
# NLTK needs additional data files (tokenizer models, stopword lists).
# These are NOT included with the pip package — must be downloaded separately.
# We download them silently on first import.

def _ensure_nltk_data() -> None:
    """Download required NLTK data if not already present."""
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", name)
            nltk.download(name, quiet=True)


_ensure_nltk_data()


# ══════════════════════════════════════════════
# DATA CLASSES FOR RESULTS
# ══════════════════════════════════════════════

@dataclass
class TextStats:
    """
    Statistics about a text, computed during preprocessing.

    📚 WHY A DATACLASS?
        We could return a dict, but dataclasses provide:
        - Type safety (IDE autocomplete, error detection)
        - Clear documentation of what fields exist
        - Immutability option (frozen=True)
        - Auto-generated __repr__ for debugging
    """
    word_count: int
    sentence_count: int
    char_count: int
    avg_word_length: float
    reading_time_minutes: float  # Based on 200 words/minute average


@dataclass
class PreprocessedText:
    """Result of text preprocessing."""
    cleaned_text: str           # Text after cleaning (for summarization)
    original_text: str          # Original text (for reference)
    stats: TextStats            # Text statistics
    sentences: list[str]        # Individual sentences
    words: list[str]            # Individual words (tokenized)


# ══════════════════════════════════════════════
# TEXT PREPROCESSOR CLASS
# ══════════════════════════════════════════════

class TextPreprocessor:
    """
    Cleans and normalizes text through a configurable pipeline.

    📚 DESIGN: Pipeline Pattern
        Each cleaning step is a small, focused function.
        The main clean() method chains them together:
            raw text → fix unicode → remove HTML → normalize whitespace → clean text

        This is better than one giant function because:
        - Each step is independently testable
        - Steps can be enabled/disabled via configuration
        - Easy to add new steps without touching existing ones

    Usage:
        >>> preprocessor = TextPreprocessor()
        >>> result = preprocessor.preprocess("  Hello   World!  \\n\\n  ")
        >>> result.cleaned_text
        'Hello World!'
        >>> result.stats.word_count
        2
    """

    # Average reading speed (words per minute)
    READING_SPEED_WPM: int = 200

    def preprocess(self, text: str) -> PreprocessedText:
        """
        Run the full preprocessing pipeline on input text.

        Pipeline steps:
            1. Fix Unicode issues (normalize, remove zero-width chars)
            2. Unescape HTML entities (&amp; → &)
            3. Remove/replace special characters
            4. Normalize whitespace
            5. Compute statistics

        Args:
            text: Raw input text.

        Returns:
            PreprocessedText with cleaned text and statistics.
        """
        logger.info("Preprocessing text (%d characters)", len(text))

        original = text

        # Step 1: Unicode normalization
        text = self.normalize_unicode(text)

        # Step 2: HTML entity unescaping
        text = self.unescape_html(text)

        # Step 3: Clean special characters
        text = self.clean_special_characters(text)

        # Step 4: Normalize whitespace
        text = self.normalize_whitespace(text)

        # Step 5: Tokenize into sentences and words
        sentences = self.tokenize_sentences(text)
        words = self.tokenize_words(text)

        # Step 6: Compute statistics
        stats = self.compute_stats(text, sentences, words)

        logger.info(
            "Preprocessing complete: %d words, %d sentences, %.1f min reading time",
            stats.word_count, stats.sentence_count, stats.reading_time_minutes,
        )

        return PreprocessedText(
            cleaned_text=text,
            original_text=original,
            stats=stats,
            sentences=sentences,
            words=words,
        )

    # ──────────────────────────────────────────
    # PIPELINE STEP 1: Unicode Normalization
    # ──────────────────────────────────────────

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize Unicode characters to a consistent form.

        📚 UNICODE NORMALIZATION FORMS:
            The character "é" can be stored two ways:
            - NFC (Composed):   U+00E9 (single codepoint)
            - NFD (Decomposed): U+0065 + U+0301 (e + combining accent)

            Both LOOK identical but have different bytes!
            NFKC normalization converts both to the same form AND
            replaces compatibility characters:
            - "ﬁ" (ligature) → "fi"
            - "½" → "1/2" (wait, actually NFKC keeps ½)
            - "Ⅳ" → "IV"

        Also removes:
            - Zero-width spaces (U+200B) — invisible characters that break processing
            - Zero-width joiners (U+200D) — used in emoji sequences
            - Byte Order Marks (U+FEFF) — file encoding markers
        """
        # NFKC: Compatibility decomposition + canonical composition
        text = unicodedata.normalize("NFKC", text)

        # Remove zero-width and invisible characters
        # 📚 These are Unicode characters that take up no visual space
        #    but can interfere with text processing and comparison.
        invisible_chars = [
            "\u200b",  # Zero-width space
            "\u200c",  # Zero-width non-joiner
            "\u200d",  # Zero-width joiner
            "\ufeff",  # Byte Order Mark (BOM)
            "\u00ad",  # Soft hyphen
        ]
        for char in invisible_chars:
            text = text.replace(char, "")

        return text

    # ──────────────────────────────────────────
    # PIPELINE STEP 2: HTML Unescaping
    # ──────────────────────────────────────────

    @staticmethod
    def unescape_html(text: str) -> str:
        """
        Convert HTML entities back to their character equivalents.

        📚 HTML ENTITIES:
            HTML uses special sequences for characters that have
            special meaning in HTML markup:
            - &amp;  → &
            - &lt;   → <
            - &gt;   → >
            - &quot; → "
            - &#39;  → '
            - &nbsp; → (non-breaking space)

            These often appear in text extracted from web pages or
            documents that were originally HTML-formatted.
        """
        return html.unescape(text)

    # ──────────────────────────────────────────
    # PIPELINE STEP 3: Special Characters
    # ──────────────────────────────────────────

    @staticmethod
    def clean_special_characters(text: str) -> str:
        """
        Replace or remove special/non-standard characters.

        📚 SMART QUOTES vs STRAIGHT QUOTES:
            Word processors often replace "straight quotes" with
            "smart quotes" (curly quotes):
            - " " → " " (left/right double quotes)
            - ' ' → ' ' (left/right single quotes)
            - — (em dash) vs -- vs - (all different!)

            We normalize these to their ASCII equivalents for consistency.

        📚 REGULAR EXPRESSIONS (re module):
            Patterns that match text:
            - r"\\s+"   matches one or more whitespace characters
            - r"[^a-zA-Z]" matches any non-letter
            - re.sub(pattern, replacement, text) — find and replace
        """
        # Smart quotes → straight quotes
        replacements = {
            "\u2018": "'",   # Left single quotation mark
            "\u2019": "'",   # Right single quotation mark (also apostrophe)
            "\u201c": '"',   # Left double quotation mark
            "\u201d": '"',   # Right double quotation mark
            "\u2013": "-",   # En dash
            "\u2014": " - ", # Em dash (add spaces for readability)
            "\u2026": "...", # Horizontal ellipsis
            "\u00a0": " ",   # Non-breaking space → regular space
            "\u2022": "- ",  # Bullet point → dash
            "\u00b7": "- ",  # Middle dot → dash
        }

        for original, replacement in replacements.items():
            text = text.replace(original, replacement)

        # Remove any remaining control characters (except newline and tab)
        # 📚 Control characters are bytes 0-31 (non-printable).
        #    We keep \n (newline, 10) and \t (tab, 9).
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        return text

    # ──────────────────────────────────────────
    # PIPELINE STEP 4: Whitespace Normalization
    # ──────────────────────────────────────────

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace to clean, consistent formatting.

        Steps:
            1. Replace tabs with spaces
            2. Collapse multiple spaces into one
            3. Collapse 3+ newlines into 2 (paragraph breaks)
            4. Strip leading/trailing whitespace from each line
            5. Strip the entire text
        """
        # Tabs → spaces
        text = text.replace("\t", " ")

        # Multiple spaces → single space (within lines)
        text = re.sub(r"[^\S\n]+", " ", text)
        # 📚 [^\S\n]+ explained:
        #    \S = non-whitespace, [^\S] = whitespace (double negative)
        #    [^\S\n] = whitespace EXCEPT newlines
        #    So this collapses spaces/tabs without affecting line breaks

        # 3+ consecutive newlines → 2 newlines (preserve paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Final strip
        return text.strip()

    # ──────────────────────────────────────────
    # TOKENIZATION METHODS
    # ──────────────────────────────────────────

    @staticmethod
    def tokenize_sentences(text: str) -> list[str]:
        """
        Split text into sentences using NLTK's Punkt tokenizer.

        📚 WHY NOT JUST SPLIT ON PERIODS?
            "Dr. Smith went to Washington D.C. He met Mr. Johnson."

            Naive split on ".": ["Dr", " Smith went to Washington D", "C", ...]
            NLTK Punkt:         ["Dr. Smith went to Washington D.C.",
                                 "He met Mr. Johnson."]

            NLTK's Punkt tokenizer is trained on large corpora and understands:
            - Abbreviations (Dr., Mr., U.S.A.)
            - Decimal numbers (3.14)
            - Ellipses (...)
            - Sentence-ending punctuation (.?!)

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        if not text.strip():
            return []

        sentences = sent_tokenize(text)
        # Filter out very short "sentences" (often artifacts)
        return [s.strip() for s in sentences if len(s.strip()) > 2]

    @staticmethod
    def tokenize_words(text: str) -> list[str]:
        """
        Split text into words using NLTK's word tokenizer.

        📚 WORD TOKENIZATION DETAILS:
            NLTK's word_tokenize uses the Penn Treebank tokenizer:
            - "I can't" → ["I", "ca", "n't"]  (splits contractions)
            - "New York-based" → ["New", "York-based"]
            - "$10.50" → ["$", "10.50"]
            - Punctuation becomes separate tokens

        Args:
            text: Input text.

        Returns:
            List of word tokens.
        """
        if not text.strip():
            return []
        return word_tokenize(text)

    # ──────────────────────────────────────────
    # STATISTICS
    # ──────────────────────────────────────────

    def compute_stats(
        self,
        text: str,
        sentences: list[str],
        words: list[str],
    ) -> TextStats:
        """
        Compute text statistics.

        📚 READING TIME CALCULATION:
            Average adult reads 200-250 words per minute.
            We use 200 WPM as a conservative estimate:
            reading_time = word_count / 200 (in minutes)
        """
        # Only count actual words (not punctuation tokens)
        content_words = [w for w in words if any(c.isalnum() for c in w)]
        word_count = len(content_words)

        avg_word_length = 0.0
        if content_words:
            avg_word_length = sum(len(w) for w in content_words) / len(content_words)

        reading_time = word_count / self.READING_SPEED_WPM if word_count > 0 else 0.0

        return TextStats(
            word_count=word_count,
            sentence_count=len(sentences),
            char_count=len(text),
            avg_word_length=round(avg_word_length, 1),
            reading_time_minutes=round(reading_time, 1),
        )

    # ──────────────────────────────────────────
    # NLP ANALYSIS UTILITIES
    # ──────────────────────────────────────────
    # These are NOT used for summarization input (models need full text).
    # They're used for keyword extraction and analysis features.

    @staticmethod
    def get_stopwords(language: str = "english") -> set[str]:
        """
        Get the set of stopwords for a language.

        📚 WHAT ARE STOPWORDS?
            The most common words in a language that carry little meaning:
            English: {"the", "is", "at", "which", "on", "a", "an", ...}

            NLTK provides stopword lists for 20+ languages.

            ⚠️ We ONLY use stopword removal for keyword extraction,
            NOT for summarization. Transformer models need full text.

        Args:
            language: Language name (default: "english").

        Returns:
            Set of stopword strings.
        """
        try:
            return set(stopwords.words(language))
        except OSError:
            logger.warning("Stopwords not available for language: %s", language)
            return set()

    @staticmethod
    def remove_stopwords(words: list[str], language: str = "english") -> list[str]:
        """
        Remove stopwords from a list of words.

        Args:
            words: List of word tokens.
            language: Language for stopword list.

        Returns:
            Filtered list without stopwords.

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.remove_stopwords(["the", "cat", "is", "happy"])
            ['cat', 'happy']
        """
        stop_words = TextPreprocessor.get_stopwords(language)
        return [w for w in words if w.lower() not in stop_words]

    @staticmethod
    def stem_words(words: list[str]) -> list[str]:
        """
        Apply Porter Stemming to a list of words.

        📚 PORTER STEMMER — HOW IT WORKS:
            Uses a series of rule-based suffix-stripping steps:
            Step 1: Plurals and -ed/-ing: "caresses" → "caress" → "car" (oops!)
            Step 2: Derivational suffixes: "relational" → "relate"
            Step 3: More suffixes: "electrical" → "electric"
            Step 4: Long suffixes: "allowance" → "allow"
            Step 5: Cleanup

            PROS: Fast, simple, well-established
            CONS: Produces non-words ("studies" → "studi")

            ⚠️ Used for analysis only, NOT for display to users.

        Args:
            words: List of word tokens.

        Returns:
            List of stemmed words.

        Example:
            >>> TextPreprocessor.stem_words(["running", "studies", "happily"])
            ['run', 'studi', 'happili']
        """
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in words]
