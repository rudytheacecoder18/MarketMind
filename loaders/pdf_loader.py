"""
loaders/pdf_loader.py
---------------------
Dual-engine PDF parser for MarketMind.

Strategy:
  1. PyPDF2      — fast, handles most machine-generated PDFs (10-Ks, transcripts).
  2. pdfplumber  — slower but more accurate for complex layouts, tables, columns.

Each page is returned as a PageRecord dict so downstream modules always have
access to the source document name and page number for citation purposes.

PageRecord schema:
  {
      "doc_name"  : str,   # original filename, e.g. "Apple_2025_10K.pdf"
      "page"      : int,   # 1-indexed page number
      "text"      : str,   # cleaned extracted text
      "char_count": int,   # len(text) after cleaning
  }
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import List, Optional

import pdfplumber
import PyPDF2

from utils.config import MIN_CHARS_PER_PAGE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pdf(
    source,
    doc_name: Optional[str] = None,
    engine: str = "auto",
) -> List[dict]:
    """
    Parse a PDF and return a list of PageRecords.

    Parameters
    ----------
    source   : file path, raw bytes, or BytesIO object.
    doc_name : display name stored in metadata (defaults to filename).
    engine   : "pypdf" | "pdfplumber" | "auto"
    """
    source, doc_name = _normalise_source(source, doc_name)

    if engine == "pypdf":
        pages = _parse_pypdf(source, doc_name)
    elif engine == "pdfplumber":
        pages = _parse_pdfplumber(source, doc_name)
    elif engine == "auto":
        pages = _parse_auto(source, doc_name)
    else:
        raise ValueError(f"Unknown engine '{engine}'. Choose: pypdf | pdfplumber | auto")

    pages = [p for p in pages if p["char_count"] >= MIN_CHARS_PER_PAGE]
    logger.info("Loaded '%s': %d pages kept (engine=%s).", doc_name, len(pages), engine)
    return pages


def load_uploaded_file(uploaded_file) -> List[dict]:
    """
    Convenience wrapper for Streamlit's UploadedFile object.

    Usage in app.py:
        pages = load_uploaded_file(st.file_uploader("Upload PDF", type="pdf"))
    """
    raw_bytes = uploaded_file.read()
    return load_pdf(source=raw_bytes, doc_name=uploaded_file.name, engine="auto")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_source(source, doc_name):
    if isinstance(source, (str, Path)):
        path = Path(source)
        doc_name = doc_name or path.name
        with open(path, "rb") as f:
            return io.BytesIO(f.read()), doc_name
    if isinstance(source, bytes):
        return io.BytesIO(source), doc_name or "document.pdf"
    if isinstance(source, io.BytesIO):
        source.seek(0)
        return source, doc_name or "document.pdf"
    raise TypeError(f"source must be a path, bytes, or BytesIO — got {type(source)}")


def _clean_text(raw: str) -> str:
    if not raw:
        return ""
    text = raw.replace("\f", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _make_record(doc_name: str, page_num: int, raw_text: str) -> dict:
    text = _clean_text(raw_text)
    return {"doc_name": doc_name, "page": page_num, "text": text, "char_count": len(text)}


def _parse_pypdf(source: io.BytesIO, doc_name: str) -> List[dict]:
    source.seek(0)
    reader = PyPDF2.PdfReader(source)
    records = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            logger.warning("PyPDF2 failed on page %d of '%s': %s", i, doc_name, exc)
            raw = ""
        records.append(_make_record(doc_name, i, raw))
    return records


def _parse_pdfplumber(source: io.BytesIO, doc_name: str) -> List[dict]:
    source.seek(0)
    records = []
    with pdfplumber.open(source) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception as exc:
                logger.warning("pdfplumber failed on page %d of '%s': %s", i, doc_name, exc)
                raw = ""
            records.append(_make_record(doc_name, i, raw))
    return records


def _parse_auto(source: io.BytesIO, doc_name: str) -> List[dict]:
    """PyPDF2 first; re-parse sparse pages with pdfplumber."""
    pypdf_records = _parse_pypdf(source, doc_name)
    sparse_pages = {r["page"] for r in pypdf_records if r["char_count"] < MIN_CHARS_PER_PAGE}

    if not sparse_pages:
        return pypdf_records

    logger.info("'%s': %d sparse page(s) — re-parsing with pdfplumber.", doc_name, len(sparse_pages))
    plumber_map = {r["page"]: r for r in _parse_pdfplumber(source, doc_name)}

    return [
        plumber_map.get(r["page"], r) if r["page"] in sparse_pages else r
        for r in pypdf_records
    ]
