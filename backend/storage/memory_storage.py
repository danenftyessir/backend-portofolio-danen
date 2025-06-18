import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .base import BaseStorage

logger = logging.getLogger(__name__)

class MemoryStorage(BaseStorage):
    """in-memory storage implementation sebagai fallback"""
    
    def __init__(self):
        self.embeddings: Optional[List[List[float]]] = None
        self.conversations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_expiry: Dict[str, datetime] = {}
        self.is_initialized = False
    
    async def initialize(self):
        """initialize memory storage"""
        try:
            # memory storage tidak perlu setup khusus
            self.is_initialized = True
            logger.info("memory storage initialized")
            
        except Exception as e:
            logger.error(f"error initializing memory storage: {e}")
            raise
    
    async def save_embeddings(self, embeddings: List[List[float]]):
        """save document embeddings ke memory"""
        try:
            self.embeddings = embeddings
            logger.info(f"saved {len(embeddings)} embeddings to memory")
            
        except Exception as e:
            logger.error(f"error saving embeddings: {e}")
            raise
    
    async def get_embeddings(self) -> Optional[List[List[float]]]:
        """get saved embeddings dari memory"""
        try:
            if self.embeddings:
                logger.info(f"retrieved {len(self.embeddings)} embeddings from memory")
            return self.embeddings
            
        except Exception as e:
            logger.error(f"error getting embeddings: {e}")
            return None
    
    async def clear_embeddings(self):
        """clear all embeddings"""
        try:
            self.embeddings = None
            logger.info("cleared embeddings from memory")
            
        except Exception as e:
            logger.error(f"error clearing embeddings: {e}")
    
    async def save_conversation(
        self,
        session_id: str,
        question: str,
        response: str,
        metadata: Dict[str, Any] = None
    ):
        """save conversation item ke memory"""
        try:
            conversation_item = {
                "session_id": session_id,
                "question": question,
                "response": response,
                "message_type": metadata.get("message_type", "general") if metadata else "general",
                "confidence_score": metadata.get("confidence_score") if metadata else None,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.conversations[session_id].append(conversation_item)
            
            # keep only last 100 conversations per session
            if len(self.conversations[session_id]) > 100:
                self.conversations[session_id] = self.conversations[session_id][-100:]
            
            logger.debug(f"saved conversation for session {session_id}")
            
        except Exception as e:
            logger.error(f"error saving conversation: {e}")
    
    async def get_conversations(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """get conversations for session"""
        try:
            conversations = self.conversations.get(session_id, [])
            return conversations[-limit:] if limit else conversations
            
        except Exception as e:
            logger.error(f"error getting conversations: {e}")
            return []
    
    async def save_session(
        self,
        session_id: str,
        session_data: Dict[str, Any]
    ):
        """save session data ke memory"""
        try:
            self.sessions[session_id] = session_data
            
            # set expiry time
            from ..config import get_settings
            settings = get_settings()
            expiry = datetime.utcnow() + timedelta(
                minutes=settings.session_timeout_minutes
            )
            self.session_expiry[session_id] = expiry
            
            logger.debug(f"saved session data for {session_id}")
            
        except Exception as e:
            logger.error(f"error saving session: {e}")
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """get session data dari memory"""
        try:
            # check if session exists
            if session_id not in self.sessions:
                return None
            
            # check if session expired
            expiry = self.session_expiry.get(session_id)
            if expiry and datetime.utcnow() > expiry:
                # cleanup expired session
                await self._delete_session(session_id)
                return None
            
            return self.sessions[session_id]
            
        except Exception as e:
            logger.error(f"error getting session: {e}")
            return None
    
    async def _delete_session(self, session_id: str):
        """delete session dari memory"""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
            if session_id in self.session_expiry:
                del self.session_expiry[session_id]
            # keep conversation history for analytics
            
        except Exception as e:
            logger.error(f"error deleting session: {e}")
    
    async def cleanup_expired_sessions(self, expire_before: datetime):
        """cleanup expired sessions"""
        try:
            expired_sessions = []
            
            for session_id, expiry in self.session_expiry.items():
                if expiry < expire_before:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                await self._delete_session(session_id)
            
            # cleanup old conversations (keep last 30 days)
            cleanup_before = datetime.utcnow() - timedelta(days=30)
            
            for session_id in list(self.conversations.keys()):
                conversations = self.conversations[session_id]
                # filter conversations yang masih dalam range
                recent_conversations = [
                    conv for conv in conversations
                    if datetime.fromisoformat(conv["created_at"]) > cleanup_before
                ]
                
                if recent_conversations:
                    self.conversations[session_id] = recent_conversations
                else:
                    del self.conversations[session_id]
            
            if expired_sessions:
                logger.info(f"cleaned up {len(expired_sessions)} expired sessions")
            
        except Exception as e:
            logger.error(f"error cleaning up expired sessions: {e}")
    
    async def get_analytics_data(self) -> Dict[str, Any]:
        """get analytics data dari memory"""
        try:
            # total conversations
            total_conversations = sum(len(convs) for convs in self.conversations.values())
            
            # active sessions (not expired)
            now = datetime.utcnow()
            active_sessions = sum(
                1 for expiry in self.session_expiry.values()
                if expiry > now
            )
            
            # conversations by day (last 7 days)
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_conversations = 0
            
            for conversations in self.conversations.values():
                for conv in conversations:
                    created_at = datetime.fromisoformat(conv["created_at"])
                    if created_at > week_ago:
                        recent_conversations += 1
            
            # popular message types
            message_types = []
            for conversations in self.conversations.values():
                for conv in conversations:
                    created_at = datetime.fromisoformat(conv["created_at"])
                    if created_at > week_ago:
                        message_types.append({
                            "message_type": conv.get("message_type", "general")
                        })
            
            return {
                "total_conversations": total_conversations,
                "active_sessions": active_sessions,
                "recent_conversations": recent_conversations,
                "message_types": message_types
            }
            
        except Exception as e:
            logger.error(f"error getting analytics data: {e}")
            return {}
    
    async def cleanup(self):
        """cleanup resources"""
        try:
            # clear all data
            self.embeddings = None
            self.conversations.clear()
            self.sessions.clear()
            self.session_expiry.clear()
            self.is_initialized = False
            
            logger.info("memory storage cleaned up")
            
        except Exception as e:
            logger.error(f"error cleaning up memory storage: {e}")
    
    async def search_conversations(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """search conversations berdasarkan query"""
        try:
            query_lower = query.lower()
            results = []
            
            for session_id, conversations in self.conversations.items():
                for conv in conversations:
                    # simple text search dalam question dan response
                    if (query_lower in conv["question"].lower() or 
                        query_lower in conv["response"].lower()):
                        results.append({
                            **conv,
                            "id": f"{session_id}_{len(results)}"  # generate id
                        })
                        
                        if len(results) >= limit:
                            break
                
                if len(results) >= limit:
                    break
            
            # sort by created_at descending
            results.sort(
                key=lambda x: x["created_at"],
                reverse=True
            )
            
            return results
            
        except Exception as e:
            logger.error(f"error searching conversations: {e}")
            return []
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """get memory usage statistics"""
        try:
            import sys
            
            embeddings_size = 0
            if self.embeddings:
                embeddings_size = sys.getsizeof(self.embeddings)
            
            conversations_size = sys.getsizeof(self.conversations)
            sessions_size = sys.getsizeof(self.sessions)
            
            return {
                "embeddings_size_bytes": embeddings_size,
                "conversations_size_bytes": conversations_size,
                "sessions_size_bytes": sessions_size,
                "total_sessions": len(self.sessions),
                "total_conversation_entries": sum(len(convs) for convs in self.conversations.values()),
                "embeddings_count": len(self.embeddings) if self.embeddings else 0
            }
            
        except Exception as e:
            logger.error(f"error getting memory usage: {e}")
            return {}