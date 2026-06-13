"""
llm/claude_client.py
--------------------
Anthropic Claude API wrapper for MarketMind.

Takes retrieved chunks + user question, builds a grounded prompt, and returns
Claude's answer. Claude is instructed to answer ONLY from the provided context
and always cite the source document and page number.
"""

from __future__ import annotations

import logging
from typing import List

import anthropic

from utils.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — ground Claude strictly in the retrieved context
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are MarketMind, an expert financial research assistant.

Your job is to answer questions about financial documents including annual reports,
earnings call transcripts, and research papers.

Rules you must follow:
1. Answer ONLY using the context provided below. Do not use outside knowledge.
2. If the answer cannot be found in the context, say exactly:
   "I could not find sufficient information in the uploaded documents."
3. Always cite your sources. After each key claim, reference the document name
   and page number like this: [Source: Apple_2025_10K.pdf, Page 72]
4. Be concise and professional. Use bullet points for lists of risks, factors, or items.
5. Never speculate or extrapolate beyond what the documents say."""


def answer(query: str, chunks: List[dict]) -> str:
    """
    Generate a grounded answer from Claude using the retrieved chunks.

    Parameters
    ----------
    query  : the user's original question.
    chunks : list of Chunk dicts from the retriever, each with
             doc_name, page, text, score keys.

    Returns
    -------
    Claude's answer as a plain string.
    """
    if not chunks:
        return "I could not find sufficient information in the uploaded documents."

    context_block = _build_context(chunks)
    user_message = f"{context_block}\n\nQuestion: {query}"

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    logger.info("Sending query to Claude (%s)...", CLAUDE_MODEL)

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    response_text = message.content[0].text
    logger.info("Claude responded (%d chars).", len(response_text))
    return response_text


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _build_context(chunks: List[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.
    """
    lines = ["CONTEXT (retrieved from uploaded documents):\n"]

    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] Document: {chunk['doc_name']} | Page: {chunk['page']}\n"
            f"{chunk['text']}\n"
        )

    return "\n".join(lines)
