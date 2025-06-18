import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import json
import asyncio
from collections import defaultdict

from ..config import Settings
from ..models import SessionInfo, ConversationItem

logger = logging.getLogger(__name__)

class SessionService:
    """service untuk mengelola user sessions"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sessions: Dict[str, SessionInfo] = {}
        self.conversation_history: Dict[str, List[ConversationItem]] = defaultdict(list)
        self.stats = {
            "total_sessions": 0,
            "total_messages": 0,
            "response_times": []
        }
        
        # cleanup task akan distart saat service diinit di main.py
        self._cleanup_task = None
    
    async def start_background_tasks(self):
        """start background tasks - dipanggil setelah event loop running"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("started session cleanup background task")
    
    async def _periodic_cleanup(self):
        """periodic cleanup untuk expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # cleanup setiap 5 menit
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"error in session cleanup: {e}")
    
    async def create_session(self, session_id: str) -> SessionInfo:
        """create new session"""
        try:
            session = SessionInfo(
                session_id=session_id,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                message_count=0,
                context={}
            )
            
            self.sessions[session_id] = session
            self.stats["total_sessions"] += 1
            
            logger.info(f"created new session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"error creating session {session_id}: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """get session by id"""
        try:
            session = self.sessions.get(session_id)
            if session:
                # check if session expired
                if self._is_session_expired(session):
                    await self._remove_session(session_id)
                    return None
                return session
            return None
            
        except Exception as e:
            logger.error(f"error getting session {session_id}: {e}")
            return None
    
    async def update_session(self, session_id: str, question: str) -> SessionInfo:
        """update session dengan new activity"""
        try:
            session = await self.get_session(session_id)
            
            if not session:
                session = await self.create_session(session_id)
            
            # update session
            session.last_activity = datetime.utcnow()
            session.message_count += 1
            
            # update stats
            self.stats["total_messages"] += 1
            
            # update context dengan question terakhir
            session.context["last_question"] = question
            session.context["last_activity"] = session.last_activity.isoformat()
            
            logger.debug(f"updated session {session_id}, message count: {session.message_count}")
            return session
            
        except Exception as e:
            logger.error(f"error updating session {session_id}: {e}")
            raise
    
    async def add_conversation_item(
        self,
        session_id: str,
        question: str,
        response: str,
        message_type: str = "general",
        confidence_score: Optional[float] = None
    ):
        """add item ke conversation history"""
        try:
            from ..models import MessageType
            
            item = ConversationItem(
                question=question,
                response=response,
                timestamp=datetime.utcnow(),
                message_type=MessageType(message_type),
                confidence_score=confidence_score
            )
            
            # add ke conversation history
            self.conversation_history[session_id].append(item)
            
            # keep only last 20 conversations per session
            if len(self.conversation_history[session_id]) > 20:
                self.conversation_history[session_id] = self.conversation_history[session_id][-20:]
            
            logger.debug(f"added conversation item to session {session_id}")
            
        except Exception as e:
            logger.error(f"error adding conversation item: {e}")
    
    async def get_conversation_history(self, session_id: str) -> List[ConversationItem]:
        """get conversation history untuk session"""
        try:
            return self.conversation_history.get(session_id, [])
        except Exception as e:
            logger.error(f"error getting conversation history: {e}")
            return []
    
    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """get session context untuk ai processing"""
        try:
            session = await self.get_session(session_id)
            conversation = await self.get_conversation_history(session_id)
            
            # build context
            context = {
                "session_id": session_id,
                "created_at": session.created_at.isoformat() if session else None,
                "message_count": session.message_count if session else 0,
                "conversation_summary": self._summarize_conversation(conversation),
                "last_topics": self._extract_recent_topics(conversation),
                "session_duration": self._calculate_session_duration(session) if session else 0
            }
            
            return context
            
        except Exception as e:
            logger.error(f"error getting session context: {e}")
            return {"session_id": session_id}
    
    def _summarize_conversation(self, conversation: List[ConversationItem]) -> Dict[str, Any]:
        """summarize conversation untuk context"""
        if not conversation:
            return {"total_exchanges": 0}
        
        # basic summary
        total_exchanges = len(conversation)
        recent_questions = [item.question for item in conversation[-3:]]
        message_types = [item.message_type.value for item in conversation]
        
        # count message types
        type_counts = {}
        for msg_type in message_types:
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
        
        return {
            "total_exchanges": total_exchanges,
            "recent_questions": recent_questions,
            "message_type_distribution": type_counts,
            "session_started": conversation[0].timestamp.isoformat() if conversation else None
        }
    
    def _extract_recent_topics(self, conversation: List[ConversationItem]) -> List[str]:
        """extract topics dari recent conversation"""
        if not conversation:
            return []
        
        # simple keyword extraction dari recent questions
        recent_questions = [item.question.lower() for item in conversation[-5:]]
        all_text = " ".join(recent_questions)
        
        # basic topic keywords
        topic_keywords = {
            "python": "python programming",
            "data science": "data science",
            "project": "project experience", 
            "algorithm": "algorithm",
            "web development": "web development",
            "music": "music preferences",
            "food": "food preferences",
            "hobi": "hobbies"
        }
        
        detected_topics = []
        for keyword, topic in topic_keywords.items():
            if keyword in all_text:
                detected_topics.append(topic)
        
        return detected_topics[:3]  # return max 3 topics
    
    def _calculate_session_duration(self, session: SessionInfo) -> int:
        """calculate session duration dalam minutes"""
        if not session:
            return 0
        
        duration = datetime.utcnow() - session.created_at
        return int(duration.total_seconds() / 60)
    
    def _is_session_expired(self, session: SessionInfo) -> bool:
        """check if session sudah expired"""
        timeout = timedelta(minutes=self.settings.session_timeout_minutes)
        return datetime.utcnow() - session.last_activity > timeout
    
    async def _cleanup_expired_sessions(self):
        """cleanup expired sessions"""
        try:
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                if self._is_session_expired(session):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                await self._remove_session(session_id)
            
            if expired_sessions:
                logger.info(f"cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            logger.error(f"error in session cleanup: {e}")
    
    async def _remove_session(self, session_id: str):
        """remove session dan conversation history"""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            if session_id in self.conversation_history:
                del self.conversation_history[session_id]
                
            logger.debug(f"removed session: {session_id}")
            
        except Exception as e:
            logger.error(f"error removing session {session_id}: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """get session statistics"""
        try:
            active_sessions = len([
                s for s in self.sessions.values() 
                if not self._is_session_expired(s)
            ])
            
            # calculate average response time
            avg_response_time = None
            if self.stats["response_times"]:
                avg_response_time = sum(self.stats["response_times"]) / len(self.stats["response_times"])
            
            # get popular questions
            all_questions = []
            for conversations in self.conversation_history.values():
                all_questions.extend([item.question for item in conversations])
            
            # simple popularity based on frequency
            question_counts = {}
            for q in all_questions:
                q_lower = q.lower()
                question_counts[q_lower] = question_counts.get(q_lower, 0) + 1
            
            top_questions = sorted(question_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_questions = [q[0] for q in top_questions]
            
            return {
                "total_sessions": self.stats["total_sessions"],
                "active_sessions": active_sessions,
                "total_messages": self.stats["total_messages"],
                "average_response_time": avg_response_time,
                "top_questions": top_questions
            }
            
        except Exception as e:
            logger.error(f"error getting session stats: {e}")
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "total_messages": 0
            }
    
    async def record_response_time(self, response_time_ms: int):
        """record response time untuk statistics"""
        try:
            self.stats["response_times"].append(response_time_ms)
            
            # keep only last 1000 response times
            if len(self.stats["response_times"]) > 1000:
                self.stats["response_times"] = self.stats["response_times"][-1000:]
                
        except Exception as e:
            logger.error(f"error recording response time: {e}")
    
    async def cleanup(self):
        """cleanup service resources"""
        try:
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("session service cleaned up")
            
        except Exception as e:
            logger.error(f"error cleaning up session service: {e}")