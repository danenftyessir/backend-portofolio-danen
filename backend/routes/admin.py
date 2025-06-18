from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging

from ..models import AdminStatsResponse
from ..config import get_settings, Settings
from ..dependencies import get_session_service, get_cache_service, get_rag_service
from ..services.session_service import SessionService
from ..services.cache_service import CacheService
from ..services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    session_service: SessionService = Depends(get_session_service),
    cache_service: CacheService = Depends(get_cache_service)
):
    """mendapatkan statistik admin"""
    try:
        stats = await session_service.get_stats()
        
        return AdminStatsResponse(
            total_sessions=stats.get("total_sessions", 0),
            active_sessions=stats.get("active_sessions", 0),
            total_messages=stats.get("total_messages", 0),
            average_response_time=stats.get("average_response_time"),
            top_questions=stats.get("top_questions", [])
        )
        
    except Exception as e:
        logger.error(f"error getting admin stats: {e}")
        return AdminStatsResponse(
            total_sessions=0,
            active_sessions=0,
            total_messages=0
        )

@router.get("/cache-stats")
async def get_cache_stats(
    cache_service: CacheService = Depends(get_cache_service)
):
    """mendapatkan statistik cache"""
    try:
        stats = await cache_service.get_stats()
        return {
            "cache_size": stats.get("cache_size", 0),
            "hit_rate": stats.get("hit_rate", 0.0),
            "total_requests": stats.get("total_requests", 0)
        }
    except Exception as e:
        logger.error(f"error getting cache stats: {e}")
        return {"error": str(e)}

@router.post("/rag/rebuild")
async def rebuild_rag_index(
    rag_service: RAGService = Depends(get_rag_service)
):
    """rebuild rag vector index"""
    try:
        if not rag_service:
            raise HTTPException(
                status_code=503,
                detail="rag service tidak tersedia"
            )
        
        await rag_service.rebuild_index()
        return {"message": "rag index berhasil direbuild"}
        
    except Exception as e:
        logger.error(f"error rebuilding rag index: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"gagal rebuild rag index: {str(e)}"
        )

@router.delete("/cache/clear")
async def clear_cache(
    cache_service: CacheService = Depends(get_cache_service)
):
    """clear semua cache"""
    try:
        await cache_service.clear_all()
        return {"message": "cache berhasil dibersihkan"}
        
    except Exception as e:
        logger.error(f"error clearing cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"gagal clear cache: {str(e)}"
        )

@router.get("/system-info")
async def get_system_info(
    settings: Settings = Depends(get_settings)
):
    """mendapatkan informasi sistem"""
    try:
        return {
            "environment": settings.environment,
            "version": settings.api_version,
            "embedding_model": settings.embedding_model,
            "openai_model": settings.openai_model,
            "cache_enabled": settings.enable_cache,
            "rate_limit": {
                "requests": settings.rate_limit_requests,
                "window": settings.rate_limit_window
            }
        }
    except Exception as e:
        logger.error(f"error getting system info: {e}")
        return {"error": str(e)}

@router.get("/health-detailed")
async def detailed_health_check(
    rag_service: RAGService = Depends(get_rag_service),
    session_service: SessionService = Depends(get_session_service),
    cache_service: CacheService = Depends(get_cache_service)
):
    """detailed health check untuk semua services"""
    try:
        health_status = {
            "rag_service": "healthy" if rag_service else "unavailable",
            "session_service": "healthy",
            "cache_service": "healthy"
        }
        
        # test rag service
        if rag_service:
            try:
                status = await rag_service.get_status()
                if status.get("status") != "healthy":
                    health_status["rag_service"] = "degraded"
            except:
                health_status["rag_service"] = "error"
        
        overall_status = "healthy"
        if any(status in ["error", "unavailable"] for status in health_status.values()):
            overall_status = "unhealthy"
        elif any(status == "degraded" for status in health_status.values()):
            overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "services": health_status,
            "timestamp": "utc_now"
        }
        
    except Exception as e:
        logger.error(f"error in detailed health check: {e}")
        return {
            "overall_status": "error",
            "error": str(e)
        }