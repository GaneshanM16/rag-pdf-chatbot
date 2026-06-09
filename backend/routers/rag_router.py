from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from backend.models.schemas import ChatRequest, ChatResponse, UploadResponse, SessionInfo
from backend.services import rag_service

router = APIRouter(prefix="/api", tags=["RAG"])


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF to create a new RAG session.
    Returns a session_id to use in subsequent /chat calls.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    max_size_mb = 20
    contents = await file.read()
    if len(contents) > max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max size is {max_size_mb}MB.")

    try:
        result = rag_service.create_session(contents, file.filename)
        return UploadResponse(
            session_id=result["session_id"],
            filename=result["filename"],
            num_chunks=result["num_chunks"],
            message=f"PDF ingested successfully into {result['num_chunks']} chunks. Ready to chat!",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question about the uploaded PDF.
    Requires a valid session_id from /upload.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = rag_service.ask_question(request.session_id, request.question)
        return ChatResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get metadata about an existing session."""
    info = rag_service.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionInfo(**info)


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free its memory."""
    deleted = rag_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session deleted successfully."}


@router.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(rag_service.sessions)}
