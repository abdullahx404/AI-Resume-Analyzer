# =============================================================================
# core/ranker.py
# Responsible for: list of score dicts - sorted, labelled pandas DataFrame.
#
# Steps:
#   1. Convert raw score results from scorer.py into a DataFrame.
#   2. Sort by final_score descending.
#   3. Assign rank (1 = best match).
#   4. Apply shortlist threshold from settings.py.
#   5. Format columns for display and CSV export.
# =============================================================================

import logging
from typing import List, Dict, Any

import pandas as pd

from config.settings import MIN_SCORE_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions (order matters for display)
# ---------------------------------------------------------------------------
_DISPLAY_COLUMNS = [
    "Rank",
    "Candidate",
    "Final Score (%)",
    "Semantic Similarity",
    "Skill Match (%)",
    "Matched Skills",
    "Missing Skills",
    "Status",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_candidates(score_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of score dicts (from scorer.compute_scores) into a
    clean, sorted DataFrame ready for display and CSV export.

    Parameters
    ----------
    score_results : List[Dict[str, Any]]
        Raw output from scorer.compute_scores().

    Returns
    -------
    pd.DataFrame
        Columns: Rank, Candidate, Final Score (%), Semantic Similarity,
                 Skill Match (%), Matched Skills, Missing Skills, Status.
        Sorted by Final Score descending (Rank 1 = best match).
        Empty DataFrame if score_results is empty.
    """
    if not score_results:
        logger.warning("rank_candidates received an empty results list.")
        return pd.DataFrame(columns=_DISPLAY_COLUMNS)

    try:
        df = _build_dataframe(score_results)
        df = _sort_and_rank(df)
        df = _apply_status(df)
        df = df[_DISPLAY_COLUMNS]

        logger.info(
            f"Ranked {len(df)} candidate(s). "
            f"Shortlisted: {(df['Status'] == 'Shortlisted').sum()}, "
            f"Rejected: {(df['Status'] == 'Rejected').sum()}"
        )
        return df

    except Exception as exc:
        logger.error(f"Ranking failed: {exc}", exc_info=True)
        return pd.DataFrame(columns=_DISPLAY_COLUMNS)


def get_shortlisted(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """Return only shortlisted candidates from a ranked DataFrame."""
    return ranked_df[ranked_df["Status"] == "Shortlisted"].reset_index(drop=True)


def get_top_n(ranked_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the top-N candidates regardless of threshold."""
    return ranked_df.head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten score dicts into a DataFrame with human-readable column names.
    Skill sets are converted to sorted, comma-separated strings.
    """
    rows = []
    for r in results:
        matched = r.get("matched_skills", set())
        missing = r.get("missing_skills", set())

        rows.append({
            "Candidate":            r.get("name", "Unknown"),
            "Final Score (%)":      r.get("final_score", 0.0),
            "Semantic Similarity":  round(r.get("semantic_score", 0.0) * 100, 1),
            "Skill Match (%)":      round(r.get("skill_score", 0.0) * 100, 1),
            "Matched Skills":       ", ".join(sorted(matched)) if matched else "None",
            "Missing Skills":       ", ".join(sorted(missing)) if missing else "None",
        })

    return pd.DataFrame(rows)


def _sort_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by Final Score descending and add Rank column."""
    df = df.sort_values("Final Score (%)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def _apply_status(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each candidate as Shortlisted or Rejected based on threshold."""
    df["Status"] = df["Final Score (%)"].apply(
        lambda score: "✅ Shortlisted" if score >= MIN_SCORE_THRESHOLD else "❌ Rejected"
    )
    return df
