# MarketMind 📊

**AI-powered Financial Research Assistant using Retrieval-Augmented Generation (RAG)**

Upload annual reports, earnings call transcripts, and research PDFs. Ask natural language questions. Get grounded answers with citations.

---

## Architecture

```
PDF Upload
    ↓
PDF Parser        (PyPDF2 + pdfplumber dual-engine)
    ↓
Text Chunker      (1000 chars, 200 overlap)
    ↓
Embeddings        (sentence-transformers: all-MiniLM-L6-v2)
    ↓
FAISS Vector Store
    ↓
User Question → Embed → Retrieve Top-5 Chunks
    ↓
Claude Sonnet API (grounded prompt, context-only)
    ↓
Answer + Citations (document name + page number)
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourname/marketmind.git
cd marketmind
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run

```bash
streamlit run app.py
```

---

## Example

**Upload:** `Apple_2025_Annual_Report.pdf`

**Question:** *What are Apple's major business risks?*

**Answer:**
> Apple identifies several key business risks:
> 1. Supply chain disruptions [Source: Apple_2025_Annual_Report.pdf, Page 81]
> 2. Foreign exchange fluctuations [Source: Apple_2025_Annual_Report.pdf, Page 72]
> 3. Regulatory challenges in international markets [Source: Apple_2025_Annual_Report.pdf, Page 85]

---

## Project Structure

```
marketmind/
├── app.py                  # Streamlit UI
├── loaders/
│   └── pdf_loader.py       # Dual-engine PDF parser (PyPDF2 + pdfplumber)
├── rag/
│   ├── chunking.py         # Sliding window text chunker
│   ├── embeddings.py       # SentenceTransformer embeddings
│   ├── vector_store.py     # FAISS index + metadata sidecar
│   └── retriever.py        # Query → top-k chunks
├── llm/
│   └── claude_client.py    # Anthropic Claude API wrapper
├── utils/
│   └── config.py           # Central configuration
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Anthropic Claude Sonnet |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS |
| PDF Parsing | PyPDF2 + pdfplumber |
| Language | Python 3.10+ |

---

## Resume Description

**MarketMind** | Python, Streamlit, Anthropic Claude API, FAISS, sentence-transformers

- Built a Retrieval-Augmented Generation (RAG) system for analyzing annual reports, earnings transcripts, and financial research documents.
- Implemented dual-engine PDF parsing, semantic chunking, vector embeddings, and FAISS-based similarity search for context-aware financial question answering.
- Integrated Anthropic Claude Sonnet with a grounded prompt architecture, returning answers with document citations and exact page numbers.
- Designed a modular pipeline (parse → chunk → embed → retrieve → generate) supporting multi-document ingestion via an interactive Streamlit interface.
