from fastapi import Depends, HTTPException, Request
from typing import Optional
import logging

from .config import get_settings, Settings
from .services.rag_service import RAGService
from .services.session_service import SessionService
from .services.cache_service import CacheService

logger = logging.getLogger(__name__)

def get_rag_service(request: Request) -> Optional[RAGService]:
    """dependency untuk mendapatkan rag service"""
    try:
        if hasattr(request.app.state, 'rag_service'):
            return request.app.state.rag_service
        return None
    except Exception as e:
        logger.error(f"error getting rag service: {e}")
        return None

def get_session_service(request: Request) -> SessionService:
    """dependency untuk mendapatkan session service"""
    try:
        if hasattr(request.app.state, 'session_service'):
            return request.app.state.session_service
        # fallback jika belum diinit
        settings = get_settings()
        return SessionService(settings)
    except Exception as e:
        logger.error(f"error getting session service: {e}")
        settings = get_settings()
        return SessionService(settings)

def get_cache_service(
    settings: Settings = Depends(get_settings)
) -> CacheService:
    """dependency untuk mendapatkan cache service"""
    return CacheService(settings)

async def verify_openai_key(
    settings: Settings = Depends(get_settings)
) -> str:
    """verify openai api key"""
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="openai api key tidak dikonfigurasi"
        )
    return settings.openai_api_key

def get_client_ip(request: Request) -> str:
    """mendapatkan client ip address untuk rate limiting"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

async def check_rate_limit(
    request: Request,
    cache_service: CacheService = Depends(get_cache_service)
) -> bool:
    """check rate limiting untuk client"""
    try:
        client_ip = get_client_ip(request)
        return await cache_service.check_rate_limit(client_ip)
    except Exception as e:
        logger.error(f"error checking rate limit: {e}")
        # jika error, allow request
        return True

def validate_session_id(session_id: Optional[str]) -> Optional[str]:
    """validate session id format"""
    if not session_id:
        return None
    
    # basic validation
    if len(session_id) < 10 or len(session_id) > 100:
        return None
    
    return session_id