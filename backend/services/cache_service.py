import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import asyncio
from cachetools import TTLCache

from ..config import Settings

logger = logging.getLogger(__name__)

class CacheService:
    """service untuk caching responses dan rate limiting"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.response_cache = TTLCache(
            maxsize=1000,
            ttl=settings.cache_ttl_seconds
        )
        self.rate_limit_cache = TTLCache(
            maxsize=10000,
            ttl=settings.rate_limit_window
        )
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def _generate_cache_key(self, question: str) -> str:
        """generate cache key dari pertanyaan"""
        # normalize question untuk caching
        normalized = question.lower().strip()
        # hash untuk key yang konsisten
        return hashlib.md5(normalized.encode()).hexdigest()
    
    async def get_response(self, question: str) -> Optional[Dict[str, Any]]:
        """get cached response untuk pertanyaan"""
        if not self.settings.enable_cache:
            return None
        
        try:
            cache_key = self._generate_cache_key(question)
            cached = self.response_cache.get(cache_key)
            
            self.stats["total_requests"] += 1
            
            if cached:
                self.stats["cache_hits"] += 1
                logger.debug(f"cache hit for question: {question[:50]}...")
                return cached
            else:
                self.stats["cache_misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"error getting cached response: {e}")
            return None
    
    async def cache_response(
        self,
        question: str,
        response: str,
        message_type: str = "general",
        related_topics: list = None,
        confidence_score: Optional[float] = None
    ):
        """cache response untuk pertanyaan"""
        if not self.settings.enable_cache:
            return
        
        try:
            cache_key = self._generate_cache_key(question)
            
            cache_data = {
                "response": response,
                "message_type": message_type,
                "related_topics": related_topics or [],
                "confidence_score": confidence_score,
                "cached_at": datetime.utcnow().isoformat()
            }
            
            self.response_cache[cache_key] = cache_data
            logger.debug(f"cached response for question: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"error caching response: {e}")
    
    async def check_rate_limit(self, client_ip: str) -> bool:
        """check rate limit untuk client ip"""
        try:
            current_time = datetime.utcnow()
            
            # get existing requests untuk ip ini
            ip_requests = self.rate_limit_cache.get(client_ip, [])
            
            # filter requests dalam time window
            time_window = timedelta(seconds=self.settings.rate_limit_window)
            cutoff_time = current_time - time_window
            
            # filter request yang masih dalam window
            recent_requests = [
                req_time for req_time in ip_requests 
                if req_time > cutoff_time
            ]
            
            # check jika sudah exceed limit
            if len(recent_requests) >= self.settings.rate_limit_requests:
                logger.warning(f"rate limit exceeded for ip: {client_ip}")
                return False
            
            # add current request
            recent_requests.append(current_time)
            self.rate_limit_cache[client_ip] = recent_requests
            
            return True
            
        except Exception as e:
            logger.error(f"error checking rate limit: {e}")
            # jika error, allow request
            return True
    
    async def get_rate_limit_status(self, client_ip: str) -> Dict[str, Any]:
        """get rate limit status untuk client"""
        try:
            current_time = datetime.utcnow()
            ip_requests = self.rate_limit_cache.get(client_ip, [])
            
            # filter recent requests
            time_window = timedelta(seconds=self.settings.rate_limit_window)
            cutoff_time = current_time - time_window
            
            recent_requests = [
                req_time for req_time in ip_requests 
                if req_time > cutoff_time
            ]
            
            remaining = max(0, self.settings.rate_limit_requests - len(recent_requests))
            
            # calculate reset time
            if recent_requests:
                oldest_request = min(recent_requests)
                reset_time = oldest_request + time_window
            else:
                reset_time = current_time
            
            return {
                "requests_made": len(recent_requests),
                "requests_limit": self.settings.rate_limit_requests,
                "requests_remaining": remaining,
                "reset_time": reset_time.isoformat(),
                "window_seconds": self.settings.rate_limit_window
            }
            
        except Exception as e:
            logger.error(f"error getting rate limit status: {e}")
            return {
                "requests_made": 0,
                "requests_limit": self.settings.rate_limit_requests,
                "requests_remaining": self.settings.rate_limit_requests,
                "reset_time": datetime.utcnow().isoformat(),
                "window_seconds": self.settings.rate_limit_window
            }
    
    async def invalidate_cache(self, question: str):
        """invalidate cache untuk specific question"""
        try:
            cache_key = self._generate_cache_key(question)
            if cache_key in self.response_cache:
                del self.response_cache[cache_key]
                logger.info(f"invalidated cache for question: {question[:50]}...")
                
        except Exception as e:
            logger.error(f"error invalidating cache: {e}")
    
    async def clear_all(self):
        """clear semua cache"""
        try:
            self.response_cache.clear()
            self.rate_limit_cache.clear()
            logger.info("cleared all caches")
            
        except Exception as e:
            logger.error(f"error clearing cache: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """get cache statistics"""
        try:
            total_requests = self.stats["total_requests"]
            cache_hits = self.stats["cache_hits"]
            cache_misses = self.stats["cache_misses"]
            
            hit_rate = 0.0
            if total_requests > 0:
                hit_rate = cache_hits / total_requests
            
            return {
                "cache_size": len(self.response_cache),
                "rate_limit_cache_size": len(self.rate_limit_cache),
                "total_requests": total_requests,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "hit_rate": hit_rate,
                "cache_enabled": self.settings.enable_cache,
                "cache_ttl_seconds": self.settings.cache_ttl_seconds
            }
            
        except Exception as e:
            logger.error(f"error getting cache stats: {e}")
            return {
                "cache_size": 0,
                "hit_rate": 0.0,
                "total_requests": 0
            }
    
    async def warm_cache(self, common_questions: list):
        """warm cache dengan common questions"""
        try:
            logger.info("warming cache with common questions...")
            
            for question in common_questions:
                # check jika sudah ada di cache
                if await self.get_response(question) is None:
                    # bisa trigger request untuk cache question ini
                    logger.debug(f"cache miss for common question: {question[:50]}...")
            
            logger.info("cache warming completed")
            
        except Exception as e:
            logger.error(f"error warming cache: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """get cache info untuk debugging"""
        try:
            return {
                "response_cache": {
                    "size": len(self.response_cache),
                    "maxsize": self.response_cache.maxsize,
                    "ttl": self.response_cache.ttl,
                    "currsize": self.response_cache.currsize if hasattr(self.response_cache, 'currsize') else len(self.response_cache)
                },
                "rate_limit_cache": {
                    "size": len(self.rate_limit_cache),
                    "maxsize": self.rate_limit_cache.maxsize,
                    "ttl": self.rate_limit_cache.ttl
                }
            }
        except Exception as e:
            logger.error(f"error getting cache info: {e}")
            return {}
    
    async def preload_responses(self, qa_pairs: list):
        """preload responses ke cache"""
        try:
            logger.info(f"preloading {len(qa_pairs)} responses to cache...")
            
            for qa_pair in qa_pairs:
                question = qa_pair.get("question", "")
                response = qa_pair.get("response", "")
                message_type = qa_pair.get("message_type", "general")
                
                if question and response:
                    await self.cache_response(
                        question=question,
                        response=response,
                        message_type=message_type
                    )
            
            logger.info("response preloading completed")
            
        except Exception as e:
            logger.error(f"error preloading responses: {e}")