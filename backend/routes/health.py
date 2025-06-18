from fastapi import APIRouter, Depends, Request
from datetime import datetime
import logging

from ..models import HealthResponse, RAGStatusResponse
from ..config import get_settings, Settings
from ..dependencies import get_rag_service
from ..services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_settings)
):
    """health check endpoint"""
    try:
        services = {
            "api": "healthy",
            "database": "healthy",  # supabase
            "cache": "healthy"
        }
        
        return HealthResponse(
            status="healthy",
            version=settings.api_version,
            services=services
        )
    except Exception as e:
        logger.error(f"health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            version=settings.api_version,
            services={"api": "error"}
        )

@router.get("/rag-status", response_model=RAGStatusResponse)
async def rag_status(
    request: Request,
    rag_service: RAGService = Depends(get_rag_service)
):
    """check rag system status"""
    try:
        if not rag_service:
            return RAGStatusResponse(
                status="unavailable",
                vector_store_size=0
            )
        
        status_info = await rag_service.get_status()
        
        return RAGStatusResponse(
            status=status_info.get("status", "unknown"),
            vector_store_size=status_info.get("vector_store_size", 0),
            last_updated=status_info.get("last_updated"),
            embedding_model=status_info.get("embedding_model")
        )
    except Exception as e:
        logger.error(f"rag status error: {e}")
        return RAGStatusResponse(
            status="error",
            vector_store_size=0
        )

@router.get("/ping")
async def ping():
    """simple ping endpoint"""
    return {
        "message": "pong",
        "timestamp": datetime.utcnow().isoformat()
    }