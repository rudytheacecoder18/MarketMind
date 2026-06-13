"""
rag/chunking.py
---------------
Splits PageRecords into overlapping text chunks for embedding.

Each chunk carries forward the doc_name and page number from its source page
so citations always know where a chunk came from.

Chunk schema:
  {
      "chunk_id"  : str,   # "{doc_name}::p{page}::c{index}"
      "doc_name"  : str,
      "page"      : int,
      "text"      : str,
      "char_count": int,
  }
"""

from __future__ import annotations

import logging
from typing import List

from utils.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def chunk_pages(pages: List[dict]) -> List[dict]:
    """
    Convert a list of PageRecords into a flat list of Chunk dicts.

    A single page may produce multiple chunks. Each chunk knows its source
    document and page number for citation.
    """
    all_chunks: List[dict] = []

    for page_record in pages:
        page_chunks = _chunk_text(
            text=page_record["text"],
            doc_name=page_record["doc_name"],
            page=page_record["page"],
        )
        all_chunks.extend(page_chunks)

    logger.info("Chunked %d pages → %d chunks.", len(pages), len(all_chunks))
    return all_chunks


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _chunk_text(text: str, doc_name: str, page: int) -> List[dict]:
    """
    Slide a window of CHUNK_SIZE characters over `text` with CHUNK_OVERLAP
    between consecutive windows.
    """
    chunks: List[dict] = []
    start = 0
    index = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "chunk_id"  : f"{doc_name}::p{page}::c{index}",
                "doc_name"  : doc_name,
                "page"      : page,
                "text"      : chunk_text,
                "char_count": len(chunk_text),
            })
            index += 1

        # Move window forward, keeping overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
