from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
import sys
from pathlib import Path

# add parent directory to path untuk relative imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from backend.config import get_settings
from backend.routes import chat, health, admin
from backend.services.rag_service import RAGService
from backend.services.ai_service import AIService
from backend.utils.logging import setup_logging

# setup logging
setup_logging()
logger = logging.getLogger(__name__)

# global services
rag_service = None
ai_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """manage application lifecycle"""
    global rag_service, ai_service
    
    # startup
    logger.info("🚀 starting danendra portfolio backend...")
    try:
        settings = get_settings()
        
        # initialize ai service
        ai_service = AIService(settings)
        app.state.ai_service = ai_service
        
        # log ai provider status
        provider_status = ai_service.get_provider_status()
        logger.info(f"🤖 primary ai provider: {settings.ai_provider}")
        logger.info(f"🤖 gemini available: {provider_status['gemini_available']}")
        logger.info(f"🤖 openai available: {provider_status['openai_available']}")
        
        # initialize rag service
        rag_service = RAGService(settings)
        await rag_service.initialize()
        app.state.rag_service = rag_service
        logger.info("✅ rag service initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ failed to initialize services: {e}")
        # jangan crash aplikasi, gunakan fallback mode
        
    yield
    
    # shutdown
    logger.info("🛑 shutting down backend...")
    if rag_service:
        await rag_service.cleanup()

# create fastapi app
app = FastAPI(
    title="danendra portfolio backend",
    description="ai-powered portfolio backend with rag system (gemini + openai)",
    version="1.0.0",
    lifespan=lifespan
)

# cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # untuk development, sesuaikan untuk production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routes
app.include_router(health.router, tags=["health"])
app.include_router(chat.router, tags=["chat"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])

@app.get("/")
async def root():
    """root endpoint dengan informasi api"""
    try:
        settings = get_settings()
        rag_status = "healthy" if hasattr(app.state, 'rag_service') and app.state.rag_service else "unavailable"
        
        # ai provider info
        ai_info = {"status": "unavailable"}
        if hasattr(app.state, 'ai_service') and app.state.ai_service:
            provider_status = app.state.ai_service.get_provider_status()
            ai_info = {
                "status": "available",
                "primary_provider": provider_status['primary_provider'],
                "gemini_available": provider_status['gemini_available'],
                "openai_available": provider_status['openai_available']
            }
        
        return {
            "message": "danendra portfolio backend api",
            "version": "1.0.0",
            "status": "running",
            "rag_status": rag_status,
            "ai_provider": ai_info,
            "environment": settings.environment,
            "endpoints": {
                "health": "/health",
                "chat": "/ask",
                "mock_chat": "/ask-mock",
                "rag_status": "/rag-status",
                "docs": "/docs"
            },
            "note": "using gemini api (free) as primary ai provider with openai fallback"
        }
    except Exception as e:
        logger.error(f"error in root endpoint: {e}")
        return {
            "message": "danendra portfolio backend api",
            "version": "1.0.0", 
            "status": "running",
            "error": "partial initialization"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True
    )