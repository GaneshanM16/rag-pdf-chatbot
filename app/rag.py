import os
from typing import Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# ── Config ────────────────────────────────────────────────────────────────────
VECTORSTORE_DIR = "vectorstore"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
MODEL_NAME = "gpt-3.5-turbo"  # swap to gpt-4o for better quality

embeddings = OpenAIEmbeddings()

# In-memory session store  {session_id: chain}
_session_chains: dict[str, Any] = {}
_ingested_files: list[str] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
    )


def _build_chain(session_id: str) -> ConversationalRetrievalChain:
    vectorstore = _get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",          # Maximal Marginal Relevance → less repetitive
        search_kwargs={"k": TOP_K},
    )
    memory = ConversationBufferWindowMemory(
        k=6,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.2)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    _session_chains[session_id] = chain
    return chain


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_pdf(file_path: str) -> int:
    """Load, chunk and embed a PDF into ChromaDB. Returns chunk count."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)

    # Tag each chunk with source filename for citation
    filename = os.path.basename(file_path)
    for chunk in chunks:
        chunk.metadata["source"] = filename

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(chunks)
    vectorstore.persist()

    _ingested_files.append(filename)

    # Invalidate cached chains so they pick up new docs
    _session_chains.clear()

    return len(chunks)


def answer_question(question: str, session_id: str) -> dict:
    """Run the question through the RAG chain and return answer + sources."""
    chain = _session_chains.get(session_id) or _build_chain(session_id)

    result = chain({"question": question})

    answer = result.get("answer", "I couldn't find an answer.")
    source_docs = result.get("source_documents", [])

    # Deduplicate sources
    sources = list(
        dict.fromkeys(
            doc.metadata.get("source", "unknown") for doc in source_docs
        )
    )

    return {"answer": answer, "sources": sources}


def list_documents() -> list[str]:
    return list(set(_ingested_files))


def clear_vectorstore():
    import shutil
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)
    _ingested_files.clear()
    _session_chains.clear()
