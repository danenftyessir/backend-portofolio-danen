from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

class BaseStorage(ABC):
    """abstract base class untuk storage implementations"""
    
    @abstractmethod
    async def initialize(self):
        """initialize storage connection"""
        pass
    
    @abstractmethod
    async def save_embeddings(self, embeddings: List[List[float]]):
        """save document embeddings"""
        pass
    
    @abstractmethod
    async def get_embeddings(self) -> Optional[List[List[float]]]:
        """get saved embeddings"""
        pass
    
    @abstractmethod
    async def clear_embeddings(self):
        """clear all embeddings"""
        pass
    
    @abstractmethod
    async def save_conversation(
        self,
        session_id: str,
        question: str,
        response: str,
        metadata: Dict[str, Any] = None
    ):
        """save conversation item"""
        pass
    
    @abstractmethod
    async def get_conversations(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """get conversations for session"""
        pass
    
    @abstractmethod
    async def save_session(
        self,
        session_id: str,
        session_data: Dict[str, Any]
    ):
        """save session data"""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """get session data"""
        pass
    
    @abstractmethod
    async def cleanup_expired_sessions(self, expire_before: datetime):
        """cleanup expired sessions"""
        pass
    
    @abstractmethod
    async def get_analytics_data(self) -> Dict[str, Any]:
        """get analytics data"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """cleanup resources"""
        pass