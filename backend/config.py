from pydantic_settings import BaseSettings
from typing import Optional
import os
from functools import lru_cache

class Settings(BaseSettings):
    """application settings dengan environment variables"""
    
    # environment
    environment: str = "development"
    log_level: str = "INFO"
    
    # api configuration
    api_title: str = "danendra portfolio backend"
    api_version: str = "1.0.0"
    port: str = "8000"
    
    # frontend configuration
    frontend_url: str = "https://portofolio-danen-frontend.vercel.app"
    
    # supabase configuration
    supabase_url: str = "https://yqdedehgjocumpjpmwrq.supabase.co"
    supabase_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxZGVkZWhnam9jdW1wanBtd3JxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTAxNjQ2OTMsImV4cCI6MjA2NTc0MDY5M30.CMvI-KJlg9FHgxUQVBHz7nkGoPcpnn0jaybdRitmWas"
    
    # gemini configuration (primary ai provider)
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"  # model gratis yang bagus
    
    # openai configuration (fallback)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    
    # ai generation settings
    max_tokens: int = 1000
    temperature: float = 0.7
    ai_provider: str = "gemini"  # primary provider: "gemini" atau "openai"
    
    # rag configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    max_context_length: int = 4000
    similarity_threshold: float = 0.7
    max_retrieved_docs: int = 5
    
    # session management
    session_timeout_minutes: int = 60
    max_sessions: int = 1000
    
    # rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    # cache configuration
    cache_ttl_seconds: int = 3600
    enable_cache: bool = True
    
    # data paths
    portfolio_data_path: str = "backend/data/portfolio.json"
    skills_data_path: str = "backend/data/skills.json"
    projects_data_path: str = "backend/data/projects.json"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    """get cached settings instance"""
    return Settings()

def is_development() -> bool:
    """check if running in development mode"""
    return get_settings().environment.lower() == "development"

def is_production() -> bool:
    """check if running in production mode"""
    return get_settings().environment.lower() == "production"