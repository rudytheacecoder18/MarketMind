"""
rag/retriever.py
----------------
Ties together the embedding model and the vector store into a single
retrieve(query) call.

The Retriever is the only module app.py needs to call at query time.
It handles:
  1. Embedding the user's question
  2. Searching FAISS for the top-k most relevant chunks
  3. Returning those chunks (with doc_name, page, text, score) for the LLM
"""

from __future__ import annotations

import logging
from typing import List

from rag.embeddings import embed_query
from rag.vector_store import VectorStore
from utils.config import TOP_K

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self._store = vector_store

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Parameters
        ----------
        query  : the user's natural language question.
        top_k  : how many chunks to return (defaults to config.TOP_K).

        Returns
        -------
        List of Chunk dicts sorted by similarity, each with keys:
          chunk_id, doc_name, page, text, char_count, score
        """
        if self._store.total_chunks == 0:
            logger.warning("Retriever called but vector store is empty.")
            return []

        query_vec = embed_query(query)
        results = self._store.search(query_vec, top_k=top_k)

        logger.info("Query: '%s' → %d chunks retrieved.", query[:60], len(results))
        return results
