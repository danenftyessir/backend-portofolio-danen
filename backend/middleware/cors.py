from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)

def setup_cors_middleware(app, settings):
    """setup cors middleware berdasarkan environment"""
    
    if settings.environment == "development":
        # development: allow all origins
        allowed_origins = ["*"]
        allow_credentials = True
    else:
        # production: restrict origins
        allowed_origins = [
            "https://your-frontend-domain.com",
            "https://www.your-frontend-domain.com",
            "https://localhost:3000",  # untuk testing
        ]
        allow_credentials = True
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-Remaining", "X-RateLimit-Reset"]
    )
    
    logger.info(f"cors middleware configured for {settings.environment}")

class CORSHeadersMiddleware:
    """custom cors headers middleware"""
    
    def __init__(self, app, settings):
        self.app = app
        self.settings = settings
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # add custom cors headers
                headers = dict(message.get("headers", []))
                
                # security headers
                headers[b"X-Content-Type-Options"] = b"nosniff"
                headers[b"X-Frame-Options"] = b"DENY"
                headers[b"X-XSS-Protection"] = b"1; mode=block"
                
                if self.settings.environment == "production":
                    headers[b"Strict-Transport-Security"] = b"max-age=31536000; includeSubDomains"
                
                message["headers"] = [(k.encode() if isinstance(k, str) else k, 
                                     v.encode() if isinstance(v, str) else v) 
                                    for k, v in headers.items()]
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)