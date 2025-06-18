import logging
import sys
from datetime import datetime
import os

def setup_logging():
    """setup aplikasi logging configuration"""
    
    # get log level dari environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # create custom formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # file handler (optional)
    file_handler = None
    if os.getenv("ENVIRONMENT", "development") == "production":
        try:
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler(
                f"logs/backend_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setFormatter(formatter)
        except Exception as e:
            print(f"failed to create file handler: {e}")
    
    # root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # clear existing handlers
    root_logger.handlers.clear()
    
    # add handlers
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
    
    # configure specific loggers
    configure_external_loggers()
    
    logging.info(f"logging configured with level: {log_level}")

def configure_external_loggers():
    """configure external library loggers"""
    
    # supabase logger
    logging.getLogger("supabase").setLevel(logging.WARNING)
    logging.getLogger("postgrest").setLevel(logging.WARNING)
    
    # openai logger
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    # httpx logger
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # sentence transformers
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    # uvicorn logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)
    
    # fastapi logger
    logging.getLogger("fastapi").setLevel(logging.INFO)

def get_logger(name: str) -> logging.Logger:
    """get logger dengan nama tertentu"""
    return logging.getLogger(name)

class ContextualLogger:
    """logger dengan context tambahan"""
    
    def __init__(self, logger_name: str, context: dict = None):
        self.logger = logging.getLogger(logger_name)
        self.context = context or {}
    
    def _format_message(self, message: str) -> str:
        """format message dengan context"""
        if self.context:
            context_str = " | ".join([f"{k}={v}" for k, v in self.context.items()])
            return f"[{context_str}] {message}"
        return message
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format_message(message), **kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(self._format_message(message), **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(self._format_message(message), **kwargs)

def create_session_logger(session_id: str) -> ContextualLogger:
    """create logger dengan session context"""
    return ContextualLogger(
        "backend.session", 
        {"session": session_id[:8]}
    )

def create_request_logger(request_id: str) -> ContextualLogger:
    """create logger dengan request context"""
    return ContextualLogger(
        "backend.request",
        {"req": request_id[:8]}
    )