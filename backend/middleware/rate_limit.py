import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from fastapi import Request, Response, HTTPException
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class RateLimitMiddleware:
    """rate limiting middleware"""
    
    def __init__(self, app, settings):
        self.app = app
        self.settings = settings
        
        # rate limit storage
        self.client_requests: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=settings.rate_limit_requests)
        )
        
        # track blocked ips
        self.blocked_ips: Dict[str, datetime] = {}
        
        # exempted endpoints
        self.exempted_paths = {'/health', '/ping', '/docs', '/openapi.json'}
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # check if path is exempted
        if request.url.path in self.exempted_paths:
            await self.app(scope, receive, send)
            return
        
        # get client ip
        client_ip = self._get_client_ip(request)
        
        # check if ip is blocked
        if await self._is_ip_blocked(client_ip):
            await self._send_rate_limit_response(
                send, 
                status_code=429,
                message="ip temporarily blocked due to rate limit violations"
            )
            return
        
        # check rate limit
        if not await self._check_rate_limit(client_ip):
            await self._send_rate_limit_response(
                send,
                status_code=429, 
                message="rate limit exceeded"
            )
            return
        
        # add rate limit headers
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                
                # add rate limit headers
                remaining = await self._get_remaining_requests(client_ip)
                reset_time = await self._get_reset_time(client_ip)
                
                headers[b"X-RateLimit-Limit"] = str(self.settings.rate_limit_requests).encode()
                headers[b"X-RateLimit-Remaining"] = str(remaining).encode()
                headers[b"X-RateLimit-Reset"] = str(int(reset_time.timestamp())).encode()
                headers[b"X-RateLimit-Window"] = str(self.settings.rate_limit_window).encode()
                
                message["headers"] = [(k.encode() if isinstance(k, str) else k,
                                     v.encode() if isinstance(v, str) else v)
                                    for k, v in headers.items()]
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
    
    def _get_client_ip(self, request: Request) -> str:
        """get client ip dengan support untuk proxy headers"""
        # check x-forwarded-for header
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # ambil ip pertama dari chain
            return forwarded.split(",")[0].strip()
        
        # check x-real-ip header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # fallback ke client host
        return request.client.host if request.client else "unknown"
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """check apakah client masih dalam rate limit"""
        try:
            current_time = time.time()
            window_start = current_time - self.settings.rate_limit_window
            
            # get request history untuk client
            requests = self.client_requests[client_ip]
            
            # remove old requests outside window
            while requests and requests[0] < window_start:
                requests.popleft()
            
            # check if limit exceeded
            if len(requests) >= self.settings.rate_limit_requests:
                logger.warning(f"rate limit exceeded for ip: {client_ip}")
                
                # track violations
                await self._track_violation(client_ip)
                return False
            
            # add current request
            requests.append(current_time)
            
            return True
            
        except Exception as e:
            logger.error(f"error checking rate limit: {e}")
            # jika error, allow request
            return True
    
    async def _track_violation(self, client_ip: str):
        """track rate limit violations"""
        # untuk ip yang repeatedly violate, bisa di-block sementara
        # simple implementation: block untuk 5 menit after 3 violations
        pass
    
    async def _is_ip_blocked(self, client_ip: str) -> bool:
        """check apakah ip di-block"""
        if client_ip in self.blocked_ips:
            block_time = self.blocked_ips[client_ip]
            if datetime.utcnow() < block_time:
                return True
            else:
                # unblock ip
                del self.blocked_ips[client_ip]
        
        return False
    
    async def _get_remaining_requests(self, client_ip: str) -> int:
        """get remaining requests untuk client"""
        try:
            current_time = time.time()
            window_start = current_time - self.settings.rate_limit_window
            
            requests = self.client_requests[client_ip]
            
            # count requests dalam current window
            current_requests = sum(1 for req_time in requests if req_time >= window_start)
            
            return max(0, self.settings.rate_limit_requests - current_requests)
            
        except Exception as e:
            logger.error(f"error getting remaining requests: {e}")
            return self.settings.rate_limit_requests
    
    async def _get_reset_time(self, client_ip: str) -> datetime:
        """get reset time untuk rate limit window"""
        try:
            requests = self.client_requests[client_ip]
            
            if requests:
                # oldest request time + window = reset time
                oldest_request = min(requests)
                reset_time = datetime.fromtimestamp(
                    oldest_request + self.settings.rate_limit_window
                )
            else:
                # no requests, reset time is now + window
                reset_time = datetime.utcnow() + timedelta(
                    seconds=self.settings.rate_limit_window
                )
            
            return reset_time
            
        except Exception as e:
            logger.error(f"error getting reset time: {e}")
            return datetime.utcnow()
    
    async def _send_rate_limit_response(self, send, status_code: int, message: str):
        """send rate limit error response"""
        response_body = {
            "error": "rate_limit_exceeded",
            "message": message,
            "status_code": status_code
        }
        
        import json
        body = json.dumps(response_body).encode()
        
        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode()],
            ]
        })
        
        await send({
            "type": "http.response.body",
            "body": body
        })
    
    def cleanup_old_data(self):
        """cleanup old rate limit data"""
        try:
            current_time = time.time()
            window_start = current_time - self.settings.rate_limit_window
            
            # cleanup old requests
            for client_ip in list(self.client_requests.keys()):
                requests = self.client_requests[client_ip]
                
                # remove old requests
                while requests and requests[0] < window_start:
                    requests.popleft()
                
                # remove empty deques
                if not requests:
                    del self.client_requests[client_ip]
            
            # cleanup expired blocks
            current_datetime = datetime.utcnow()
            expired_blocks = [
                ip for ip, block_time in self.blocked_ips.items()
                if current_datetime >= block_time
            ]
            
            for ip in expired_blocks:
                del self.blocked_ips[ip]
            
            logger.debug(f"cleaned up rate limit data for {len(expired_blocks)} ips")
            
        except Exception as e:
            logger.error(f"error cleaning up rate limit data: {e}")

class DynamicRateLimitMiddleware(RateLimitMiddleware):
    """dynamic rate limiting berdasarkan endpoint dan user behavior"""
    
    def __init__(self, app, settings):
        super().__init__(app, settings)
        
        # per-endpoint limits
        self.endpoint_limits = {
            "/ask": 50,  # more restrictive untuk ai endpoints
            "/ask-mock": 100,
            "/health": 1000,  # very permissive untuk health checks
        }
        
        # trusted ips (contoh: monitoring systems)
        self.trusted_ips = set()
    
    async def _check_rate_limit(self, client_ip: str, request: Request = None) -> bool:
        """enhanced rate limit check dengan per-endpoint limits"""
        
        # trusted ips bypass rate limiting
        if client_ip in self.trusted_ips:
            return True
        
        # get endpoint-specific limit
        endpoint = request.url.path if request else ""
        limit = self.endpoint_limits.get(endpoint, self.settings.rate_limit_requests)
        
        try:
            current_time = time.time()
            window_start = current_time - self.settings.rate_limit_window
            
            # create endpoint-specific key
            key = f"{client_ip}:{endpoint}"
            
            requests = self.client_requests[key]
            
            # remove old requests
            while requests and requests[0] < window_start:
                requests.popleft()
            
            if len(requests) >= limit:
                logger.warning(f"rate limit exceeded for {key} (limit: {limit})")
                return False
            
            requests.append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"error in dynamic rate limit check: {e}")
            return True