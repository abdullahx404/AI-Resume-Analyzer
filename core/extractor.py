# =============================================================================
# core/extractor.py  (skill_extractor)
# Responsible for: cleaned text - set of matched technical skills.
#
# Strategy:
#   1. Single-word skills  - checked via token-level set lookup  (O(1))
#   2. Multi-word skills   - checked via sliding n-gram window   (O(n))
#
# Works on BOTH resume text and job description so scorer.py can
# compute the intersection (matched skills) and difference (missing skills).
# =============================================================================

import logging
from typing import Set, List

from data.skills_db import SKILLS_SET, SKILL_DISPLAY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-compute multi-word skills for ngram matching
# (single-word skills are handled faster via token lookup)
# ---------------------------------------------------------------------------
_MULTI_WORD_SKILLS: List[str] = sorted(
    [s for s in SKILLS_SET if " " in s],
    key=lambda s: -len(s.split()),   # longest first → more specific match wins
)
_SINGLE_WORD_SKILLS: Set[str] = {s for s in SKILLS_SET if " " not in s}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_skills(cleaned_text: str) -> Set[str]:
    """
    Extract all recognised technical skills from pre-cleaned text.

    Parameters
    ----------
    cleaned_text : str
        Output of preprocess.clean_text() — lowercase, no punctuation.

    Returns
    -------
    Set[str]
        Set of display-case skill names found in the text.
        Example: {"Python", "Machine Learning", "PostgreSQL"}
    """
    if not cleaned_text or not cleaned_text.strip():
        logger.warning("extract_skills received empty text.")
        return set()

    try:
        tokens: List[str] = cleaned_text.split()
        found: Set[str] = set()

        # ── Multi-word skill matching (sliding window) ──────────────────────
        found |= _match_ngrams(cleaned_text)

        # ── Single-word skill matching (token set intersection) ──────────────
        found |= _match_single_tokens(tokens)

        logger.info(f"Extracted {len(found)} skill(s): {found}")
        return found

    except Exception as exc:
        logger.error(f"Skill extraction failed: {exc}", exc_info=True)
        return set()


def skill_overlap(resume_skills: Set[str], jd_skills: Set[str]) -> dict:
    """
    Compute the overlap statistics between a resume's skills and the JD skills.

    Returns
    -------
    dict with keys:
        matched   : Set[str]  — skills present in both
        missing   : Set[str]  — JD skills absent from resume
        score     : float     — overlap ratio [0, 1];  0 if jd_skills is empty
    """
    if not jd_skills:
        return {"matched": set(), "missing": set(), "score": 0.0}

    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    score   = len(matched) / len(jd_skills)

    return {
        "matched": matched,
        "missing": missing,
        "score":   round(score, 4),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _match_ngrams(text: str) -> Set[str]:
    """
    Slide a window over the text to find multi-word skills.
    Uses pre-sorted list (longest first) to prefer specific matches.
    """
    found: Set[str] = set()
    for skill_lower in _MULTI_WORD_SKILLS:
        if skill_lower in text:
            # Retrieve display-case version before adding
            display = SKILL_DISPLAY.get(skill_lower, skill_lower.title())
            found.add(display)
    return found


def _match_single_tokens(tokens: List[str]) -> Set[str]:
    """
    Check each token against the single-word skill set.
    Token-level check avoids false positives from substring matching.
    """
    found: Set[str] = set()
    for token in tokens:
        if token in _SINGLE_WORD_SKILLS:
            display = SKILL_DISPLAY.get(token, token.capitalize())
            found.add(display)
    return found
