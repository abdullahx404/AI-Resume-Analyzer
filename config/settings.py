# =============================================================================
# config/settings.py
# Global configuration constants for the Resume Screening System.
# Centralizes all tuneable parameters so the rest of the codebase
# never contains magic numbers or hardcoded strings.
# =============================================================================

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# Hugging Face model identifier used by SentenceTransformer.
# all-MiniLM-L6-v2: 384-dim, ~90 MB, fast inference, strong semantic quality.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Scoring weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
SEMANTIC_WEIGHT = 0.70   # Cosine similarity between resume and JD embeddings
SKILL_WEIGHT    = 0.30   # Jaccard-style skill overlap score

# ---------------------------------------------------------------------------
# Ranking threshold
# ---------------------------------------------------------------------------
# Candidates whose final score (0-100) fall below this value are tagged
# as "Rejected" in the output table; above → "Shortlisted"
MIN_SCORE_THRESHOLD = 50.0

# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE_MB   = 10        # Maximum accepted PDF size in megabytes
TEMP_DIR           = "tmp_uploads"   # Relative path for temporary PDF storage

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
TOP_N_CANDIDATES = 10   # How many candidates to highlight in the UI summary
