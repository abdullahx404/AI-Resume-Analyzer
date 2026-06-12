# =============================================================================
# core/preprocess.py
# Responsible for: raw text - clean, normalised text.
#
# Pipeline (in order):
#   1. Lowercase
#   2. Remove URLs and email addresses
#   3. Remove punctuation and special characters
#   4. Tokenize
#   5. Remove NLTK stopwords + custom resume noise words
#   6. Lemmatize tokens
#   7. Rejoin into a single clean string
#
# Both resume text and job description pass through this same pipeline
# before any embedding or skill extraction.
# =============================================================================

import re
import logging
import string
from functools import lru_cache
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure required NLTK data is present (safe to call repeatedly)
# ---------------------------------------------------------------------------
_REQUIRED_NLTK = ["stopwords", "wordnet", "punkt", "punkt_tab"]

def _download_nltk_data() -> None:
    for resource in _REQUIRED_NLTK:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)

_download_nltk_data()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LEMMATIZER = WordNetLemmatizer()

# Additional words common in resumes that add no signal for matching
_CUSTOM_STOPWORDS: set[str] = {
    "experience", "work", "years", "year", "month", "months",
    "responsible", "responsibilities", "ability", "skills", "skill",
    "knowledge", "use", "using", "used", "good", "excellent",
    "strong", "including", "etc", "e", "g", "also", "may", "must",
    "well", "able", "job", "role", "candidate", "required", "requirement",
    "requirements", "company", "team", "position", "apply", "application",
    "resume", "cv", "looking", "seeking",
}

@lru_cache(maxsize=1)
def _get_stop_words() -> set[str]:
    """Load and cache the combined stopword set."""
    base = set(stopwords.words("english"))
    return base | _CUSTOM_STOPWORDS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Run the full preprocessing pipeline on a raw text string.

    Parameters
    ----------
    text : str
        Raw text from a PDF resume or job description paste.

    Returns
    -------
    str
        Space-joined cleaned tokens ready for embedding and skill extraction.
        Returns empty string if input is empty or only whitespace.
    """
    if not text or not text.strip():
        logger.warning("clean_text received empty or whitespace-only input.")
        return ""

    try:
        text = _lowercase(text)
        text = _remove_urls(text)
        text = _remove_emails(text)
        text = _remove_punctuation(text)
        tokens = _tokenize(text)
        tokens = _remove_stopwords(tokens)
        tokens = _lemmatize(tokens)
        return " ".join(tokens)

    except Exception as exc:
        logger.error(f"Preprocessing failed: {exc}", exc_info=True)
        return text  # Fallback: return lowercased unprocessed text


def get_tokens(text: str) -> List[str]:
    """
    Return the list of clean tokens (rather than a joined string).
    Useful for skill extraction which iterates over tokens.
    """
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []


# ---------------------------------------------------------------------------
# Private step functions
# ---------------------------------------------------------------------------

def _lowercase(text: str) -> str:
    return text.lower()


def _remove_urls(text: str) -> str:
    """Strip http/https URLs."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def _remove_emails(text: str) -> str:
    """Strip email addresses."""
    return re.sub(r"\S+@\S+", " ", text)


def _remove_punctuation(text: str) -> str:
    """
    Replace punctuation and non-alphanumeric characters with spaces.
    Keeps internal hyphens in compound words (e.g. 'real-time') as spaces.
    """
    # Replace hyphens/slashes with space to split compound tokens
    text = re.sub(r"[-/]", " ", text)
    # Remove all remaining punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    """Word-tokenize using NLTK's punkt tokenizer."""
    return word_tokenize(text)


def _remove_stopwords(tokens: List[str]) -> List[str]:
    stop_words = _get_stop_words()
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def _lemmatize(tokens: List[str]) -> List[str]:
    """Reduce tokens to their base form (e.g. 'learning' → 'learn')."""
    return [_LEMMATIZER.lemmatize(t) for t in tokens]
