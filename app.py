"""
app.py
------
MarketMind — AI-powered Financial Research Assistant.

Streamlit UI that wires together the full RAG pipeline:
  Upload PDF → Parse → Chunk → Embed → FAISS → Retrieve → Claude → Answer + Citations
"""

import logging

import streamlit as st

from loaders.pdf_loader import load_uploaded_file
from rag.chunking import chunk_pages
from rag.embeddings import embed_chunks
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from llm.claude_client import answer
from utils.config import TOP_K

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MarketMind",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever(st.session_state.vector_store)

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []      # list of filenames already ingested

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # list of {"role": "user"|"assistant", "content": str}

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []       # chunks from the most recent retrieval

# ---------------------------------------------------------------------------
# Sidebar — document management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 MarketMind")
    st.caption("AI-powered Financial Research Assistant")
    st.divider()

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload annual reports, transcripts, or research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_docs]

        if new_files:
            for uploaded_file in new_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Parse → Chunk → Embed → Store
                        pages = load_uploaded_file(uploaded_file)
                        chunks = chunk_pages(pages)
                        embeddings = embed_chunks(chunks)
                        st.session_state.vector_store.add(chunks, embeddings)
                        st.session_state.uploaded_docs.append(uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} — {len(chunks)} chunks indexed")
                    except Exception as e:
                        st.error(f"❌ Failed to process {uploaded_file.name}: {e}")

    st.divider()
    st.subheader("Indexed Documents")
    if st.session_state.uploaded_docs:
        for doc in st.session_state.uploaded_docs:
            st.markdown(f"📄 {doc}")
        st.caption(f"Total chunks: {st.session_state.vector_store.total_chunks}")
    else:
        st.info("No documents uploaded yet.")

    st.divider()
    if st.button("🗑️ Clear All Documents", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.uploaded_docs = []
        st.session_state.chat_history = []
        st.session_state.last_sources = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------

st.title("MarketMind")
st.caption("Ask questions about your uploaded financial documents.")

# Render chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input(
    "Ask a question about your documents...",
    disabled=st.session_state.vector_store.total_chunks == 0,
)

if query:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Retrieve + answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            retrieved_chunks = st.session_state.retriever.retrieve(query, top_k=TOP_K)
            response = answer(query, retrieved_chunks)
            st.session_state.last_sources = retrieved_chunks

        st.markdown(response)

        # Citations expander
        if retrieved_chunks:
            with st.expander(f"📎 Sources ({len(retrieved_chunks)} chunks retrieved)", expanded=False):
                for i, chunk in enumerate(retrieved_chunks, start=1):
                    st.markdown(
                        f"**[{i}] {chunk['doc_name']} — Page {chunk['page']}** "
                        f"*(similarity: {chunk['score']:.2f})*"
                    )
                    st.markdown(
                        f"> {chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}"
                    )
                    if i < len(retrieved_chunks):
                        st.divider()

    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Hint when no documents loaded
if st.session_state.vector_store.total_chunks == 0:
    st.info("👈 Upload a PDF in the sidebar to get started.")
