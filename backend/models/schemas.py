from pydantic import BaseModel
from typing import Optional

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    num_chunks: int
    message: str

class SessionInfo(BaseModel):
    session_id: str
    filename: str
    num_chunks: int
    chat_history_length: int
