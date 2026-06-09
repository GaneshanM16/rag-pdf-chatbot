# 🤖 PDF Q&A Chatbot — RAG with LangChain + FastAPI

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload PDF documents and ask questions in natural language. Built with **FastAPI**, **LangChain**, **ChromaDB**, and **OpenAI**.

---

## ✨ Features

- 📄 **PDF Ingestion** — Upload one or more PDFs via drag-and-drop or file picker
- 🔍 **Smart Chunking** — RecursiveCharacterTextSplitter with configurable chunk size & overlap
- 🧠 **Vector Search** — ChromaDB with OpenAI embeddings + MMR retrieval
- 💬 **Conversational Memory** — Last 6 turns of conversation context per session
- 📎 **Source Citations** — Every answer shows which PDF file(s) it drew from
- 🌐 **Web UI** — Dark-mode single-page chat interface, zero frontend dependencies
- 🐳 **Docker Ready** — One-command deployment

---

## 🏗️ Architecture

```
User Browser
    │
    ▼
FastAPI (app/main.py)
    ├── POST /upload  ──► PyPDFLoader ──► TextSplitter ──► Embeddings ──► ChromaDB
    ├── POST /chat    ──► ConversationalRetrievalChain ──► OpenAI GPT ──► Answer
    ├── GET  /documents
    └── DELETE /documents

ChromaDB (local vectorstore/)
OpenAI API (embeddings + chat completion)
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd rag-pdf-chatbot
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY
```

### 2. Run Locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open **http://localhost:8000**

### 3. Run with Docker

```bash
docker-compose up --build
```

---

## 🔧 Configuration

Edit `app/rag.py` to tune the pipeline:

```python
CHUNK_SIZE    = 800     # Characters per chunk
CHUNK_OVERLAP = 100     # Overlap between chunks
TOP_K         = 4       # Chunks retrieved per query
MODEL_NAME    = "gpt-3.5-turbo"   # or "gpt-4o" for better reasoning
```

---

## 📡 API Reference

**POST /upload** — Upload and index a PDF (multipart/form-data)

**POST /chat** — Ask a question
```json
{ "question": "What is the main conclusion?", "session_id": "user-123" }
```

**GET /documents** — List all ingested documents

**DELETE /documents** — Clear the entire knowledge base

---

## 📁 Project Structure

```
rag-pdf-chatbot/
├── app/
│   ├── main.py          # FastAPI routes
│   └── rag.py           # RAG pipeline (ingest, embed, retrieve, generate)
├── frontend/
│   └── index.html       # Single-file chat UI
├── uploads/             # Uploaded PDFs (gitignored)
├── vectorstore/         # ChromaDB data (gitignored)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI |
| RAG orchestration | LangChain |
| Vector database | ChromaDB (local) |
| Embeddings | OpenAI text-embedding-ada-002 |
| LLM | OpenAI GPT-3.5-turbo / GPT-4o |
| PDF parsing | PyPDF |

---

## 🚢 Deployment

Push to Railway, Render, or Fly.io. Set `OPENAI_API_KEY` as an environment variable. The Dockerfile handles everything else.

For production, consider swapping ChromaDB for a hosted vector DB (Pinecone, Qdrant, Weaviate) and adding auth to `/upload`.

---

## 📝 License

MIT
