from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
    """tipe pesan dalam chat"""
    GREETING = "greeting"
    PROFESSIONAL = "professional"
    PERSONAL = "personal"
    FEEDBACK = "feedback"
    GENERAL = "general"

class ChatRequest(BaseModel):
    """request model untuk chat endpoint"""
    question: str = Field(..., min_length=1, max_length=1000, description="pertanyaan user")
    session_id: Optional[str] = Field(None, description="session id untuk context")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=[], description="riwayat percakapan")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or v.strip() == "":
            raise ValueError("pertanyaan tidak boleh kosong")
        return v.strip()

class ChatResponse(BaseModel):
    """response model untuk chat endpoint"""
    response: str = Field(..., description="jawaban ai")
    session_id: str = Field(..., description="session id")
    message_type: MessageType = Field(..., description="tipe pesan")
    related_topics: Optional[List[str]] = Field(default=[], description="topik terkait")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="skor kepercayaan")
    processing_time_ms: Optional[int] = Field(None, description="waktu pemrosesan dalam ms")
    
class MockChatResponse(BaseModel):
    """response model untuk mock chat (offline mode)"""
    response: str = Field(..., description="jawaban mock")
    session_id: str = Field(..., description="session id")
    is_mock: bool = Field(True, description="indicator mock response")

class HealthResponse(BaseModel):
    """response model untuk health check"""
    status: str = Field(..., description="status aplikasi")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="versi aplikasi")
    services: Dict[str, str] = Field(..., description="status services")

class RAGStatusResponse(BaseModel):
    """response model untuk rag status"""
    status: str = Field(..., description="status rag system")
    vector_store_size: Optional[int] = Field(None, description="jumlah dokumen dalam vector store")
    last_updated: Optional[datetime] = Field(None, description="terakhir update")
    embedding_model: Optional[str] = Field(None, description="model embedding yang digunakan")

class SessionInfo(BaseModel):
    """informasi session user"""
    session_id: str = Field(..., description="id session")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = Field(default=0, description="jumlah pesan dalam session")
    context: Optional[Dict[str, Any]] = Field(default={}, description="context session")

class ConversationItem(BaseModel):
    """item dalam riwayat percakapan"""
    question: str = Field(..., description="pertanyaan user")
    response: str = Field(..., description="jawaban ai")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_type: MessageType = Field(..., description="tipe pesan")
    confidence_score: Optional[float] = Field(None, description="skor kepercayaan")

class SuggestedFollowupsResponse(BaseModel):
    """response untuk suggested followups"""
    suggested_followups: List[str] = Field(..., description="daftar pertanyaan lanjutan")
    session_id: str = Field(..., description="session id")

class PortfolioDocument(BaseModel):
    """dokumen portfolio dalam knowledge base"""
    id: str = Field(..., description="id dokumen")
    category: str = Field(..., description="kategori dokumen")
    title: str = Field(..., description="judul dokumen")
    content: str = Field(..., description="konten dokumen")
    keywords: List[str] = Field(..., description="kata kunci")
    embedding: Optional[List[float]] = Field(None, description="vector embedding")

class ErrorResponse(BaseModel):
    """response untuk error"""
    error: str = Field(..., description="pesan error")
    detail: Optional[str] = Field(None, description="detail error")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AdminStatsResponse(BaseModel):
    """response untuk admin statistics"""
    total_sessions: int = Field(..., description="total session")
    active_sessions: int = Field(..., description="session aktif")
    total_messages: int = Field(..., description="total pesan")
    average_response_time: Optional[float] = Field(None, description="rata-rata waktu respons")
    top_questions: Optional[List[str]] = Field(None, description="pertanyaan populer")