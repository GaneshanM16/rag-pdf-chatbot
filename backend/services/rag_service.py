import uuid
import tempfile
import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

from backend.config import settings


# In-memory session store: session_id -> session data
sessions: dict[str, dict] = {}


def create_session(pdf_bytes: bytes, filename: str) -> dict:
    """
    Ingest a PDF and create a new RAG session.
    Returns session metadata.
    """
    session_id = str(uuid.uuid4())

    # Write PDF to a temp file (PyPDFLoader requires a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        # 1. Load PDF pages
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # 2. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks: list[Document] = splitter.split_documents(pages)

        # 3. Embed & index with FAISS
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        vector_store = FAISS.from_documents(chunks, embeddings)

        # 4. Build conversational QA chain
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_kwargs={"k": settings.max_retrieval_docs}
            ),
            memory=memory,
            return_source_documents=True,
            verbose=False,
        )

        # 5. Store session
        sessions[session_id] = {
            "session_id": session_id,
            "filename": filename,
            "num_chunks": len(chunks),
            "qa_chain": qa_chain,
            "chat_history": [],
        }

        return {
            "session_id": session_id,
            "filename": filename,
            "num_chunks": len(chunks),
        }

    finally:
        os.unlink(tmp_path)


def ask_question(session_id: str, question: str) -> dict:
    """
    Run a question through the RAG chain for the given session.
    Returns answer + source page references.
    """
    if session_id not in sessions:
        raise ValueError(f"Session '{session_id}' not found. Please upload a PDF first.")

    session = sessions[session_id]
    qa_chain: ConversationalRetrievalChain = session["qa_chain"]

    result = qa_chain.invoke({"question": question})

    answer: str = result["answer"]
    source_docs: list[Document] = result.get("source_documents", [])

    # Deduplicate source references
    sources = []
    seen = set()
    for doc in source_docs:
        page = doc.metadata.get("page", "?")
        ref = f"{session['filename']} — page {page + 1}"
        if ref not in seen:
            sources.append(ref)
            seen.add(ref)

    # Keep a simple chat history log
    session["chat_history"].append({"role": "user", "content": question})
    session["chat_history"].append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "sources": sources,
        "session_id": session_id,
    }


def get_session_info(session_id: str) -> Optional[dict]:
    session = sessions.get(session_id)
    if not session:
        return None
    return {
        "session_id": session_id,
        "filename": session["filename"],
        "num_chunks": session["num_chunks"],
        "chat_history_length": len(session["chat_history"]),
    }


def delete_session(session_id: str) -> bool:
    if session_id in sessions:
        del sessions[session_id]
        return True
    return False
