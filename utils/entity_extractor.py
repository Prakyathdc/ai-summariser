"""
Named Entity Recognition (NER) Module
========================================

📚 WHAT THIS MODULE DOES:
    Identifies and classifies named entities in text:
    - PERSON: "Elon Musk", "Albert Einstein"
    - ORG:    "Google", "United Nations"
    - GPE:    "India", "New York" (Geo-Political Entities = countries, cities, states)
    - DATE:   "January 2024", "last Tuesday"
    - MONEY:  "$10 million", "€500"
    - And more (PRODUCT, EVENT, LAW, LANGUAGE, etc.)

📚 HOW NER WORKS IN spaCy:
    ═══════════════════════
    spaCy uses a trained neural network (CNN-based) for NER:

    1. TOKENIZATION: Split text into tokens
    2. EMBEDDING: Each token → vector representation
    3. CONTEXT: CNN layers capture surrounding context
    4. CLASSIFICATION: Each token is classified as:
       - B-PERSON: Beginning of a person entity
       - I-PERSON: Inside (continuation) of a person entity
       - O: Outside any entity (not a named entity)

    This is called BIO tagging (Begin, Inside, Outside).

    Example:
    "Barack Obama visited India"
    Barack  → B-PERSON
    Obama   → I-PERSON
    visited → O
    India   → B-GPE

📚 spaCy MODELS:
    - en_core_web_sm: Small (12 MB), fast, less accurate
    - en_core_web_md: Medium (40 MB), good balance
    - en_core_web_lg: Large (560 MB), most accurate
    We use _sm for speed in a web app. Users can upgrade if needed.

📚 LIBRARY: spaCy
    Industrial-strength NLP library. Faster than NLTK for most tasks.
    Provides: tokenization, POS tagging, NER, dependency parsing, lemmatization.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Optional
import logging

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)

# Module-level spaCy model (loaded once, reused)
_nlp: Optional[Language] = None


def _get_nlp() -> Language:
    """
    Load spaCy model (singleton pattern — loaded once, reused).

    📚 SINGLETON PATTERN:
        We load the spaCy model only ONCE and reuse it for all NER calls.
        Model loading is expensive (~1-2 seconds) — we don't want to reload
        it every time someone clicks "Summarize".

        This module-level caching is simpler than a full Singleton class
        and is the Pythonic way to handle this.

    📚 spacy.load() / spacy.blank():
        - spacy.load("en_core_web_sm"): Loads a pre-trained model
        - spacy.blank("en"): Creates an empty model (no NER, no POS tagging)
        If the model isn't installed, we fall back to blank and warn the user.
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' loaded successfully")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
            _nlp = spacy.blank("en")
    return _nlp


@dataclass
class NamedEntity:
    """
    A single named entity found in the text.

    Attributes:
        text: The entity text (e.g., "Barack Obama")
        label: Entity type (e.g., "PERSON")
        description: Human-readable description of the entity type
        start_char: Character offset where entity starts in the original text
        end_char: Character offset where entity ends
    """
    text: str
    label: str
    description: str
    start_char: int
    end_char: int


# Entity label descriptions for display
ENTITY_DESCRIPTIONS: dict[str, str] = {
    "PERSON": "Person name",
    "NORP": "Nationality, religion, or political group",
    "FAC": "Building, airport, highway, etc.",
    "ORG": "Organization, company, agency",
    "GPE": "Country, city, or state",
    "LOC": "Non-GPE location (mountain, river, etc.)",
    "PRODUCT": "Product name (vehicles, foods, etc.)",
    "EVENT": "Named event (war, sports event, etc.)",
    "WORK_OF_ART": "Title of book, song, etc.",
    "LAW": "Named legal document",
    "LANGUAGE": "Named language",
    "DATE": "Absolute or relative date",
    "TIME": "Time expression",
    "PERCENT": "Percentage",
    "MONEY": "Monetary value",
    "QUANTITY": "Measurement (weight, distance, etc.)",
    "ORDINAL": "First, second, third, etc.",
    "CARDINAL": "Numeral (not fitting other types)",
}


def extract_entities(text: str, max_length: int = 100000) -> list[NamedEntity]:
    """
    Extract named entities from text using spaCy.

    📚 nlp(text) — WHAT HAPPENS INSIDE:
        1. Tokenizer splits text into tokens
        2. Tagger assigns part-of-speech tags (noun, verb, etc.)
        3. Parser builds dependency tree (subject → verb → object)
        4. NER component identifies and classifies entities
        5. Returns a Doc object with all annotations

    📚 doc.ents:
        A tuple of Span objects, each representing a named entity.
        Each span has: .text, .label_, .start_char, .end_char

    Args:
        text: Input text to analyze.
        max_length: Maximum text length (spaCy default is 1M characters).

    Returns:
        List of NamedEntity objects, deduplicated.
    """
    if not text or len(text.strip()) < 5:
        return []

    nlp = _get_nlp()

    # Truncate very long texts (spaCy can be slow on huge inputs)
    if len(text) > max_length:
        logger.warning("Text truncated from %d to %d chars for NER", len(text), max_length)
        text = text[:max_length]

    # Process text through spaCy pipeline
    doc = nlp(text)

    # Extract entities
    entities: list[NamedEntity] = []
    seen: set[tuple[str, str]] = set()  # Track (text, label) to deduplicate

    for ent in doc.ents:
        key = (ent.text.strip(), ent.label_)
        if key in seen:
            continue
        seen.add(key)

        entities.append(NamedEntity(
            text=ent.text.strip(),
            label=ent.label_,
            description=ENTITY_DESCRIPTIONS.get(ent.label_, "Unknown entity type"),
            start_char=ent.start_char,
            end_char=ent.end_char,
        ))

    logger.info("Extracted %d unique named entities", len(entities))
    return entities


def get_entity_summary(entities: list[NamedEntity]) -> dict[str, list[str]]:
    """
    Group entities by type for display.

    Returns:
        Dict mapping entity type labels to lists of entity texts.

    Example:
        >>> summary = get_entity_summary(entities)
        >>> summary
        {'PERSON': ['Barack Obama', 'Elon Musk'], 'ORG': ['Google', 'NASA']}
    """
    summary: dict[str, list[str]] = {}
    for entity in entities:
        label = f"{entity.label} ({entity.description})"
        if label not in summary:
            summary[label] = []
        if entity.text not in summary[label]:
            summary[label].append(entity.text)

    return summary


def get_entity_counts(entities: list[NamedEntity]) -> dict[str, int]:
    """
    Count entities by type for charts.

    Returns:
        Dict mapping entity type labels to counts.
    """
    counter = Counter(ent.label for ent in entities)
    return dict(counter.most_common())
