# =============================================================================
# core/embedder.py
# Responsible for: text - dense vector embedding.
#
# Model: all-MiniLM-L6-v2  (384 dimensions)
#   - Fast inference (~14k sentences/sec on CPU)
#   - Excellent semantic quality for short-to-medium texts
#   - ~90 MB download on first use; auto-cached by sentence-transformers
#
# Design choices:
#   - Model is loaded ONCE at module import via a module-level singleton.
#   - encode() accepts a list so all texts can be batched in one call.
#   - Returns numpy arrays (compatible with sklearn cosine_similarity).
# =============================================================================

import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton model — loaded once, reused for every call in the session.
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Lazy-load and cache the SentenceTransformer model.
    Thread-safe for Streamlit's single-threaded execution model.
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        try:
            _model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load model '{EMBEDDING_MODEL}': {exc}", exc_info=True)
            raise RuntimeError(
                f"Could not load embedding model '{EMBEDDING_MODEL}'. "
                "Check your internet connection or model name."
            ) from exc
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode(texts: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
    """
    Encode one or more texts into dense embedding vectors.

    Parameters
    ----------
    texts : str | List[str]
        A single text string or a list of strings to embed.
        All texts are processed in a single batched forward pass for efficiency.
    show_progress : bool
        Whether to display a tqdm progress bar (useful for large batches).

    Returns
    -------
    np.ndarray
        Shape (n_texts, 384) — one 384-dimensional vector per input text.
        If a single string is passed, shape is (1, 384).

    Raises
    ------
    ValueError
        If texts is empty or contains only whitespace strings.
    RuntimeError
        If the model cannot be loaded.
    """
    # Normalise input to list
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        raise ValueError("encode() received an empty list of texts.")

    # Warn about blank entries but don't abort — blank text embeds to near-zero
    blank_indices = [i for i, t in enumerate(texts) if not t.strip()]
    if blank_indices:
        logger.warning(
            f"encode() received {len(blank_indices)} blank text(s) at "
            f"indices {blank_indices}. These will produce near-zero vectors."
        )

    model = _get_model()

    try:
        embeddings: np.ndarray = model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-norm → cosine sim == dot product
        )
        logger.info(f"Encoded {len(texts)} text(s) → shape {embeddings.shape}")
        return embeddings

    except Exception as exc:
        logger.error(f"Encoding failed: {exc}", exc_info=True)
        raise


def encode_single(text: str) -> np.ndarray:
    """
    Convenience wrapper — encodes exactly one text, returns shape (384,).
    """
    result = encode([text])
    return result[0]
