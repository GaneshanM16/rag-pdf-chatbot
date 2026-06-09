from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.rag import ingest_pdf, answer_question, list_documents, clear_vectorstore
import os

app = FastAPI(title="PDF Q&A Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
def root():
    return FileResponse("frontend/index.html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    chunks = ingest_pdf(file_path)
    return {
        "message": f"✅ '{file.filename}' ingested successfully.",
        "chunks_created": chunks,
        "filename": file.filename,
    }


@app.post("/chat", response_model=QuestionResponse)
def chat(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = answer_question(request.question, request.session_id)
    return result


@app.get("/documents")
def get_documents():
    return {"documents": list_documents()}


@app.delete("/documents")
def clear_documents():
    clear_vectorstore()
    return {"message": "✅ All documents cleared."}


@app.get("/health")
def health():
    return {"status": "ok"}
