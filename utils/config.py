"""
utils/config.py
---------------
Central configuration for MarketMind.
All tunable parameters live here — change once, applies everywhere.
"""

import os

# --- Anthropic ---
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = "claude-sonnet-4-6"
MAX_TOKENS: int = 1024

# --- Embeddings ---
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384          # dimension for all-MiniLM-L6-v2

# --- Chunking ---
CHUNK_SIZE: int = 1000            # characters per chunk
CHUNK_OVERLAP: int = 200          # overlap between consecutive chunks

# --- Retrieval ---
TOP_K: int = 5                    # number of chunks to retrieve per query

# --- PDF Parsing ---
MIN_CHARS_PER_PAGE: int = 20      # pages below this are re-parsed with pdfplumber
