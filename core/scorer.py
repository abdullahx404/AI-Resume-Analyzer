# =============================================================================
# core/scorer.py
# Responsible for: computing the final match score for each resume.
#
# Formula:
#   Final Score = (SEMANTIC_WEIGHT × cosine_similarity)
#               + (SKILL_WEIGHT    × skill_overlap_score)
#   Scaled to [0, 100] for display.
#
# Inputs come from:
#   - embedder.py   → numpy vectors (for cosine similarity)
#   - extractor.py  → skill sets    (for skill overlap)
#   - settings.py   → weight constants
# =============================================================================

import logging
from typing import List, Set, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import SEMANTIC_WEIGHT, SKILL_WEIGHT
from core.extractor import skill_overlap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_scores(
    resume_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
    resume_skills_list: List[Set[str]],
    jd_skills: Set[str],
    candidate_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Compute the final score for every resume against the job description.

    Parameters
    ----------
    resume_embeddings : np.ndarray
        Shape (n_resumes, 384) — one embedding per resume.
    jd_embedding : np.ndarray
        Shape (1, 384) or (384,) — job description embedding.
    resume_skills_list : List[Set[str]]
        One skill set per resume (display-case skill names).
    jd_skills : Set[str]
        Skill set extracted from the job description.
    candidate_names : List[str]
        File names or candidate labels, one per resume.

    Returns
    -------
    List[Dict[str, Any]]
        One dict per resume with keys:
            name            : str
            semantic_score  : float  [0, 1]
            skill_score     : float  [0, 1]
            final_score     : float  [0, 100]
            matched_skills  : Set[str]
            missing_skills  : Set[str]
    """
    # Ensure jd_embedding is 2-D for sklearn
    jd_vec = np.array(jd_embedding).reshape(1, -1)

    results: List[Dict[str, Any]] = []

    for idx, (name, resume_vec, resume_skills) in enumerate(
        zip(candidate_names, resume_embeddings, resume_skills_list)
    ):
        try:
            # ── Semantic similarity ─────────────────────────────────────────
            r_vec   = np.array(resume_vec).reshape(1, -1)
            sem_sim = float(cosine_similarity(r_vec, jd_vec)[0][0])
            # Clamp to [0, 1] — cosine can be slightly negative for unrelated docs
            sem_sim = max(0.0, min(1.0, sem_sim))

            # ── Skill overlap ───────────────────────────────────────────────
            overlap      = skill_overlap(resume_skills, jd_skills)
            skill_score  = overlap["score"]          # already in [0, 1]
            matched      = overlap["matched"]
            missing      = overlap["missing"]

            # ── Weighted final score ────────────────────────────────────────
            raw_score    = (SEMANTIC_WEIGHT * sem_sim) + (SKILL_WEIGHT * skill_score)
            final_score  = round(raw_score * 100, 2)   # scale to [0, 100]

            logger.debug(
                f"[{name}] sem={sem_sim:.3f}  skill={skill_score:.3f}  "
                f"final={final_score:.1f}"
            )

            results.append({
                "name":           name,
                "semantic_score": round(sem_sim, 4),
                "skill_score":    round(skill_score, 4),
                "final_score":    final_score,
                "matched_skills": matched,
                "missing_skills": missing,
            })

        except Exception as exc:
            logger.error(f"Scoring failed for '{name}': {exc}", exc_info=True)
            # Include the candidate with zeroed scores so the table stays complete
            results.append({
                "name":           name,
                "semantic_score": 0.0,
                "skill_score":    0.0,
                "final_score":    0.0,
                "matched_skills": set(),
                "missing_skills": jd_skills.copy(),
            })

    return results


def score_single(
    resume_embedding: np.ndarray,
    jd_embedding: np.ndarray,
    resume_skills: Set[str],
    jd_skills: Set[str],
) -> Dict[str, Any]:
    """
    Score a single resume (convenience wrapper around compute_scores).
    Returns the same dict structure as compute_scores entries.
    """
    results = compute_scores(
        resume_embeddings=np.array([resume_embedding]),
        jd_embedding=jd_embedding,
        resume_skills_list=[resume_skills],
        jd_skills=jd_skills,
        candidate_names=["candidate"],
    )
    return results[0]
