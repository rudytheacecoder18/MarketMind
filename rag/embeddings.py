"""
rag/embeddings.py
-----------------
Generates dense vector embeddings for text chunks using SentenceTransformers.

The model (all-MiniLM-L6-v2) is loaded once and cached — loading it on every
query would be very slow.

Returns a numpy float32 array of shape (num_chunks, EMBEDDING_DIM) which
is passed directly into the FAISS index.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once per process
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the cached embedding model, loading it on first call."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


def embed_chunks(chunks: List[dict]) -> np.ndarray:
    """
    Embed a list of Chunk dicts.

    Parameters
    ----------
    chunks : list of dicts with at least a "text" key.

    Returns
    -------
    np.ndarray of shape (len(chunks), EMBEDDING_DIM), dtype float32.
    """
    if not chunks:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    texts = [c["text"] for c in chunks]
    model = get_model()

    logger.info("Embedding %d chunks...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise for cosine similarity via inner product
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns
    -------
    np.ndarray of shape (1, EMBEDDING_DIM), dtype float32.
    """
    model = get_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.astype(np.float32)
