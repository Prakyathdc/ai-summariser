"""
Keyword Extractor Module — TF-IDF Based Keyword Extraction
============================================================

📚 WHAT THIS MODULE DOES:
    Extracts the most important keywords and keyphrases from text.
    Keywords are displayed in the UI to help users understand the main topics.

📚 HOW TF-IDF WORKS (Term Frequency–Inverse Document Frequency):
    ═══════════════════════════════════════════════════════════════

    TF-IDF is a statistical measure of how important a word is to a document.
    It combines two scores:

    1. TF (Term Frequency): How often does the word appear in this document?
       TF("cat", document) = count("cat") / total_words
       A word that appears 10 times in a 100-word doc → TF = 0.10

    2. IDF (Inverse Document Frequency): How rare is the word across documents?
       IDF("cat") = log(total_documents / documents_containing_"cat")
       A word in every document → IDF ≈ 0 (not informative)
       A word in only 1 document → IDF is high (very informative)

    3. TF-IDF = TF × IDF
       Words that are FREQUENT in this document but RARE overall get high scores.
       "the" → high TF, very low IDF → low TF-IDF (common everywhere)
       "quantum" → moderate TF, high IDF → high TF-IDF (domain-specific)

    📚 FOR SINGLE DOCUMENTS:
        Since we often have just one document, we use a trick:
        Split the document into sentences and treat each sentence as a "mini-document".
        This gives us IDF across sentences — words appearing in few sentences score higher.

📚 ALTERNATIVE: RAKE (Rapid Automatic Keyword Extraction)
    - Uses co-occurrence of words within phrases
    - Doesn't need multiple documents
    - We implement a simplified version as a fallback

📚 LIBRARY: scikit-learn's TfidfVectorizer
    - Handles tokenization, stopword removal, and TF-IDF calculation
    - Supports n-grams (single words, bigrams, trigrams)
    - Highly optimized (sparse matrices for memory efficiency)
"""

from collections import Counter
from typing import Optional
import logging
import re

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logger = logging.getLogger(__name__)


def extract_keywords_tfidf(
    text: str,
    top_n: int = 15,
    ngram_range: tuple[int, int] = (1, 2),
) -> list[dict[str, float]]:
    """
    Extract keywords using TF-IDF across sentences.

    📚 HOW THIS WORKS:
        1. Split text into sentences (each sentence = a "document")
        2. Build a TF-IDF matrix: rows=sentences, columns=words
        3. Sum TF-IDF scores across all sentences for each word
        4. Return the top-N words with highest total scores

    📚 ngram_range=(1, 2):
        - (1, 1): Only single words ("machine", "learning")
        - (1, 2): Single words AND bigrams ("machine learning")
        - (1, 3): Up to trigrams ("natural language processing")
        Bigrams often capture more meaningful concepts than single words.

    Args:
        text: Input text.
        top_n: Number of keywords to extract.
        ngram_range: Range of n-gram sizes to consider.

    Returns:
        List of dicts with 'keyword' and 'score' keys, sorted by score descending.

    Example:
        >>> keywords = extract_keywords_tfidf("Machine learning is a field of AI...")
        >>> keywords[0]
        {'keyword': 'machine learning', 'score': 0.85}
    """
    if not text or len(text.split()) < 5:
        logger.warning("Text too short for keyword extraction")
        return []

    # Split into sentences (each sentence = a "document" for TF-IDF)
    sentences = sent_tokenize(text)

    if len(sentences) < 2:
        # If only one sentence, split on clauses/chunks instead
        sentences = re.split(r"[,;:]", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if len(sentences) < 2:
        # Fallback: use simple word frequency
        return _extract_keywords_frequency(text, top_n)

    try:
        # Create TF-IDF vectorizer
        # 📚 TfidfVectorizer parameters:
        #   - ngram_range: (min_n, max_n) for word n-grams
        #   - stop_words: Remove common English words
        #   - max_features: Limit vocabulary size (memory efficiency)
        #   - max_df: Ignore words appearing in >80% of sentences (too common)
        #   - min_df: Ignore words appearing in <1 sentence (too rare / typos)
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            stop_words="english",
            max_features=1000,
            max_df=0.85,
            min_df=1,
        )

        # Fit and transform: sentences → TF-IDF matrix
        # 📚 Shape: (num_sentences, vocabulary_size)
        #    Each cell = TF-IDF score of word j in sentence i
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Get feature names (the actual words/phrases)
        feature_names = vectorizer.get_feature_names_out()

        # Sum TF-IDF scores across all sentences for each word
        # 📚 .toarray() converts sparse matrix to dense NumPy array
        #    .sum(axis=0) sums down columns (across sentences)
        scores = tfidf_matrix.toarray().sum(axis=0)

        # Create (keyword, score) pairs and sort
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)

        # Normalize scores to 0-1 range
        max_score = keyword_scores[0][1] if keyword_scores else 1.0
        results = [
            {
                "keyword": kw,
                "score": round(float(score / max_score), 3),
            }
            for kw, score in keyword_scores[:top_n]
        ]

        logger.info("Extracted %d keywords via TF-IDF", len(results))
        return results

    except Exception as e:
        logger.warning("TF-IDF extraction failed: %s, falling back to frequency", e)
        return _extract_keywords_frequency(text, top_n)


def _extract_keywords_frequency(text: str, top_n: int = 15) -> list[dict[str, float]]:
    """
    Simple frequency-based keyword extraction (fallback).

    Used when text is too short for TF-IDF to work well.
    Simply counts word frequencies after removing stopwords.
    """
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())

    # Filter: keep only alphabetic words that aren't stopwords
    content_words = [
        w for w in words
        if w.isalpha() and w not in stop_words and len(w) > 2
    ]

    # Count frequencies
    counter = Counter(content_words)
    most_common = counter.most_common(top_n)

    if not most_common:
        return []

    max_count = most_common[0][1]
    return [
        {
            "keyword": word,
            "score": round(count / max_count, 3),
        }
        for word, count in most_common
    ]


def extract_keyphrases(text: str, top_n: int = 10) -> list[str]:
    """
    Extract multi-word keyphrases using a simple RAKE-like approach.

    📚 RAKE (Rapid Automatic Keyword Extraction):
        1. Split text at stopwords and punctuation → candidate phrases
        2. Score each phrase based on:
           - Word frequency
           - Word degree (how many other words it co-occurs with in phrases)
           - Score = degree / frequency
        3. Return top-scoring phrases

    This is a simplified version that uses phrase frequency.

    Args:
        text: Input text.
        top_n: Number of keyphrases to extract.

    Returns:
        List of keyphrase strings.
    """
    stop_words = set(stopwords.words("english"))

    # Split at stopwords and punctuation to get candidate phrases
    # 📚 This regex splits on: stopwords surrounded by word boundaries, or punctuation
    words = text.lower().split()
    phrases = []
    current_phrase = []

    for word in words:
        # Remove punctuation from word
        clean_word = re.sub(r"[^\w]", "", word)

        if clean_word and clean_word not in stop_words and len(clean_word) > 2:
            current_phrase.append(clean_word)
        else:
            if len(current_phrase) >= 2:  # Only multi-word phrases
                phrases.append(" ".join(current_phrase))
            current_phrase = []

    # Don't forget the last phrase
    if len(current_phrase) >= 2:
        phrases.append(" ".join(current_phrase))

    # Count phrase frequencies
    counter = Counter(phrases)
    top_phrases = [phrase for phrase, _ in counter.most_common(top_n)]

    logger.info("Extracted %d keyphrases", len(top_phrases))
    return top_phrases
