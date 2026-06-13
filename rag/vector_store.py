"""
rag/vector_store.py
-------------------
FAISS vector store with a parallel metadata sidecar.

FAISS stores float32 vectors and integer indices only — it has no concept of
metadata. We maintain a plain Python list (`_metadata`) where position i
corresponds to FAISS internal index i. This lets us map any retrieved index
back to its original chunk dict (doc_name, page, text, chunk_id).

The store is held in memory for a session. For persistence across restarts,
call save() / load().
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import List, Optional

import faiss
import numpy as np

from utils.config import EMBEDDING_DIM

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        # IndexFlatIP = exact inner-product search (cosine similarity when
        # embeddings are L2-normalised, which embed_chunks does for us)
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._metadata: List[dict] = []   # parallel list — index i ↔ FAISS id i

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, chunks: List[dict], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the store.

        Parameters
        ----------
        chunks     : list of Chunk dicts (must include doc_name, page, text).
        embeddings : float32 array of shape (len(chunks), EMBEDDING_DIM).
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) must match."
            )

        self._index.add(embeddings)
        self._metadata.extend(chunks)
        logger.info("Added %d chunks. Total in store: %d.", len(chunks), len(self._metadata))

    def clear(self) -> None:
        """Remove all vectors and metadata."""
        self._index.reset()
        self._metadata.clear()
        logger.info("Vector store cleared.")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int) -> List[dict]:
        """
        Return the top_k most similar chunks for a query vector.

        Parameters
        ----------
        query_vector : float32 array of shape (1, EMBEDDING_DIM).
        top_k        : number of results to return.

        Returns
        -------
        List of Chunk dicts, ordered by similarity (highest first).
        Each dict gets an extra "score" key (inner product / cosine similarity).
        """
        if self._index.ntotal == 0:
            logger.warning("Search called on empty vector store.")
            return []

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self._metadata[idx])   # copy so we don't mutate stored data
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    @property
    def total_chunks(self) -> int:
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save FAISS index and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(self._metadata, f)
        logger.info("Vector store saved to '%s'.", directory)

    def load(self, directory: str) -> None:
        """Load FAISS index and metadata from disk."""
        index_path = os.path.join(directory, "index.faiss")
        meta_path  = os.path.join(directory, "metadata.pkl")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"No saved store found in '{directory}'.")
        self._index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self._metadata = pickle.load(f)
        logger.info("Vector store loaded from '%s'. Chunks: %d.", directory, len(self._metadata))
