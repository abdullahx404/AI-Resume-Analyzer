# =============================================================================
# core/parser.py  (pdf_extractor)
# Responsible for: PDF - clean raw text extraction.
#
# Accepts either a file path (str / Path) or a file-like object
# (e.g. Streamlit's UploadedFile).  Returns a plain string.
#
# Handles:
#   - Multi-page PDFs
#   - Encrypted / password-protected PDFs   <- uses doc.is_encrypted API
#   - Corrupted PDFs                         <- fitz.FileDataError
#   - Empty PDFs (no extractable text)       <- fitz.EmptyFileError (>=1.24)
#   - Non-PDF file types
#
# Compatibility note:
#   fitz.PasswordError was REMOVED in PyMuPDF >=1.18.  The correct modern
#   pattern is to check doc.is_encrypted after open() and call
#   doc.authenticate(password).  This file uses that pattern exclusively.
# =============================================================================

import io
import logging
from pathlib import Path
from typing import Union

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(source: Union[str, Path, bytes, io.IOBase]) -> str:
    """
    Extract and return all text from a PDF document.

    Parameters
    ----------
    source : str | Path | bytes | file-like object
        The PDF to process.  Accepts a filesystem path, raw bytes,
        or any readable binary stream (e.g. Streamlit UploadedFile).

    Returns
    -------
    str
        Concatenated text from every page, separated by newlines.
        Returns an empty string if extraction fails or the file is empty.

    Raises
    ------
    ValueError
        If the source is not a recognised PDF.
    """
    try:
        doc = _open_document(source)

        # Encryption check (replaces the removed fitz.PasswordError)
        # PyMuPDF >=1.18 removed fitz.PasswordError entirely.
        # The correct approach is: open succeeds but doc.is_encrypted == True.
        # doc.authenticate("") tries an empty password (covers many Word exports).
        # Returns 0 on failure, non-zero on success.
        if doc.is_encrypted:
            unlocked = doc.authenticate("")   # try empty password first
            if not unlocked:
                logger.error(
                    "PDF is password-protected and could not be unlocked. "
                    "Provide the password via doc.authenticate(password)."
                )
                doc.close()
                return ""
            logger.info("Encrypted PDF unlocked with empty password.")

        text = _extract_all_pages(doc)
        page_count = doc.page_count
        doc.close()

        if not text.strip():
            logger.warning("PDF opened successfully but contains no extractable text.")
            return ""

        logger.info(f"Extracted {len(text)} characters from {page_count} page(s).")
        return text

    except fitz.EmptyFileError:
        # Raised in PyMuPDF ≥1.24 for zero-byte files
        logger.error("PDF file is empty (0 bytes).")
        return ""

    except fitz.FileDataError as exc:
        # Corrupted or structurally invalid PDF binary
        logger.error(f"Corrupted or invalid PDF — {exc}")
        return ""

    except Exception as exc:
        logger.error(f"Unexpected error during PDF extraction: {exc}", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _open_document(source: Union[str, Path, bytes, io.IOBase]) -> fitz.Document:
    """
    Open a fitz.Document from various input types.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")
        return fitz.open(str(path))

    elif isinstance(source, bytes):
        # Raw bytes stream (e.g. read from Streamlit UploadedFile)
        return fitz.open(stream=source, filetype="pdf")

    elif hasattr(source, "read"):
        # File-like object (Streamlit UploadedFile, BytesIO, etc.)
        raw = source.read()
        if not raw:
            raise ValueError("Uploaded file is empty (0 bytes).")
        return fitz.open(stream=raw, filetype="pdf")

    else:
        raise TypeError(f"Unsupported source type: {type(source)}")


def _extract_all_pages(doc: fitz.Document) -> str:
    """
    Iterate every page of the document and concatenate the extracted text.
    Uses PyMuPDF's 'text' extraction mode which preserves reading order.
    """
    pages_text = []
    for page_num, page in enumerate(doc):
        try:
            page_text = page.get_text("text")   # plain text, reading order
            if page_text.strip():
                pages_text.append(page_text)
        except Exception as exc:
            # Skip unreadable pages rather than aborting the whole document
            logger.warning(f"Could not read page {page_num + 1}: {exc}")
            continue

    return "\n".join(pages_text)
