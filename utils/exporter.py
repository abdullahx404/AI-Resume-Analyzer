# =============================================================================
# utils/exporter.py
# Responsible for: converting a ranked DataFrame to a downloadable CSV.
#
# Called by app.py's st.download_button.
# Returns raw bytes so Streamlit can serve the file in-memory
# without writing anything to disk.
# =============================================================================

import io
import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Serialise a ranked candidates DataFrame to UTF-8 CSV bytes.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ranker.rank_candidates() — already sorted and labelled.

    Returns
    -------
    bytes
        UTF-8 encoded CSV content with BOM so Excel opens it cleanly.
        Returns empty bytes if df is None or empty.
    """
    if df is None or df.empty:
        logger.warning("to_csv_bytes received an empty DataFrame.")
        return b""

    try:
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, encoding="utf-8")
        csv_str = buffer.getvalue()
        # UTF-8 BOM ensures Excel auto-detects encoding correctly
        return ("\ufeff" + csv_str).encode("utf-8")

    except Exception as exc:
        logger.error(f"CSV export failed: {exc}", exc_info=True)
        return b""


def get_export_filename() -> str:
    """
    Generate a timestamped filename for the CSV download.
    Example: resume_screening_results_20240503_142300.csv
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"resume_screening_results_{timestamp}.csv"
