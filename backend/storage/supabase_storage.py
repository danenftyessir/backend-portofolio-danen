import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from .base import BaseStorage
from ..config import Settings

logger = logging.getLogger(__name__)

class SupabaseStorage(BaseStorage):
    """supabase storage implementation"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        self.is_initialized = False
        
    async def initialize(self):
        """initialize supabase connection"""
        try:
            # import supabase di dalam method untuk error handling yang lebih baik
            try:
                from supabase import create_client
            except ImportError:
                raise Exception("supabase package not installed. run: pip install supabase")
            
            # validate settings
            if not self.settings.supabase_url or not self.settings.supabase_key:
                raise Exception("supabase_url or supabase_key not configured")
            
            logger.info(f"🔗 connecting to supabase: {self.settings.supabase_url}")
            
            # create client dengan parameter yang benar
            self.client = create_client(
                supabase_url=self.settings.supabase_url,
                supabase_key=self.settings.supabase_key
            )
            
            # test connection
            await self._test_connection()
            
            # ensure tables
            await self._ensure_tables_exist()
            
            self.is_initialized = True
            logger.info("✅ supabase storage initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ supabase storage initialization failed: {e}")
            logger.info("   - check your supabase url and key")
            logger.info("   - ensure supabase project is active")
            logger.info("   - run the table creation sql script")
            raise
    
    async def _test_connection(self):
        """test supabase connection"""
        try:
            # test dengan query sederhana
            response = self.client.table("conversations").select("id").limit(1).execute()
            logger.info("✅ supabase connection test successful")
            
            # log jumlah existing data
            if response.data:
                logger.info(f"📊 found {len(response.data)} existing conversations")
            else:
                logger.info("📊 conversations table is empty (ready for new data)")
                
        except Exception as e:
            error_msg = str(e).lower()
            
            if "relation" in error_msg and "does not exist" in error_msg:
                logger.warning("⚠️ supabase tables not found - need to run setup sql")
                logger.info("   run the sql script in supabase dashboard to create tables")
                raise Exception("supabase tables not created yet")
            elif "permission denied" in error_msg or "rls" in error_msg:
                logger.warning("⚠️ supabase permission denied - check row level security")
                raise Exception("supabase permission denied - check rls policies")
            else:
                logger.error(f"❌ supabase connection failed: {e}")
                raise
    
    async def _ensure_tables_exist(self):
        """ensure required tables exist dengan basic check"""
        try:
            # coba akses setiap table untuk memastikan ada
            tables_to_check = ["conversations", "sessions", "embeddings"]
            
            for table_name in tables_to_check:
                try:
                    response = self.client.table(table_name).select("*").limit(1).execute()
                    logger.debug(f"✅ table '{table_name}' exists")
                except Exception as e:
                    if "does not exist" in str(e).lower():
                        logger.warning(f"❌ table '{table_name}' not found")
                        raise Exception(f"table {table_name} not created yet")
                    else:
                        # other errors like permission might be ok for now
                        logger.debug(f"⚠️ table '{table_name}' check: {e}")
            
            logger.info("✅ all required tables exist")
            
        except Exception as e:
            logger.warning(f"⚠️ table check failed: {e}")
            # don't raise here, let it continue but warn user
    
    async def save_embeddings(self, embeddings: List[List[float]]):
        """save document embeddings ke supabase"""
        try:
            if not self.client or not self.is_initialized:
                raise Exception("supabase client not initialized")
            
            # clear existing embeddings first
            delete_response = self.client.table("embeddings").delete().neq("id", 0).execute()
            logger.info("🗑️ cleared existing embeddings")
            
            # prepare new embeddings data
            embeddings_data = []
            for i, embedding in enumerate(embeddings):
                embeddings_data.append({
                    "document_id": f"doc_{i}",
                    "embedding": embedding,
                    "created_at": datetime.utcnow().isoformat()
                })
            
            # batch insert new embeddings
            if embeddings_data:
                response = self.client.table("embeddings").insert(embeddings_data).execute()
                logger.info(f"💾 saved {len(embeddings)} embeddings to supabase")
            
        except Exception as e:
            logger.error(f"❌ error saving embeddings: {e}")
            # don't raise - fallback to memory
    
    async def get_embeddings(self) -> Optional[List[List[float]]]:
        """get saved embeddings dari supabase"""
        try:
            if not self.client or not self.is_initialized:
                return None
            
            response = self.client.table("embeddings")\
                .select("embedding")\
                .order("id")\
                .execute()
            
            if response.data:
                embeddings = [row["embedding"] for row in response.data]
                logger.info(f"📥 retrieved {len(embeddings)} embeddings from supabase")
                return embeddings
            
            logger.info("📥 no embeddings found in supabase")
            return None
            
        except Exception as e:
            logger.error(f"❌ error getting embeddings: {e}")
            return None
    
    async def clear_embeddings(self):
        """clear all embeddings"""
        try:
            if not self.client or not self.is_initialized:
                return
            
            response = self.client.table("embeddings").delete().neq("id", 0).execute()
            logger.info("🗑️ cleared all embeddings from supabase")
            
        except Exception as e:
            logger.error(f"❌ error clearing embeddings: {e}")
    
    async def save_conversation(
        self,
        session_id: str,
        question: str,
        response: str,
        metadata: Dict[str, Any] = None
    ):
        """save conversation item ke supabase"""
        try:
            if not self.client or not self.is_initialized:
                return
            
            conversation_data = {
                "session_id": session_id,
                "question": question,
                "response": response,
                "message_type": metadata.get("message_type", "general") if metadata else "general",
                "confidence_score": metadata.get("confidence_score") if metadata else None,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat()
            }
            
            response = self.client.table("conversations").insert(conversation_data).execute()
            logger.debug(f"💬 saved conversation for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ error saving conversation: {e}")
    
    async def get_conversations(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """get conversations for session"""
        try:
            if not self.client or not self.is_initialized:
                return []
            
            response = self.client.table("conversations")\
                .select("*")\
                .eq("session_id", session_id)\
                .order("created_at", desc=False)\
                .limit(limit)\
                .execute()
            
            conversations = response.data or []
            logger.debug(f"📥 retrieved {len(conversations)} conversations for session {session_id}")
            return conversations
            
        except Exception as e:
            logger.error(f"❌ error getting conversations: {e}")
            return []
    
    async def save_session(
        self,
        session_id: str,
        session_data: Dict[str, Any]
    ):
        """save session data ke supabase"""
        try:
            if not self.client or not self.is_initialized:
                return
            
            expires_at = datetime.utcnow() + timedelta(
                minutes=self.settings.session_timeout_minutes
            )
            
            session_record = {
                "session_id": session_id,
                "session_data": session_data,
                "expires_at": expires_at.isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # upsert session (insert or update)
            response = self.client.table("sessions")\
                .upsert(session_record, on_conflict="session_id")\
                .execute()
            
            logger.debug(f"👤 saved session data for {session_id}")
            
        except Exception as e:
            logger.error(f"❌ error saving session: {e}")
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """get session data dari supabase"""
        try:
            if not self.client or not self.is_initialized:
                return None
            
            response = self.client.table("sessions")\
                .select("*")\
                .eq("session_id", session_id)\
                .single()\
                .execute()
            
            if response.data:
                # check if session expired
                expires_at_str = response.data["expires_at"]
                expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
                
                if datetime.utcnow() > expires_at.replace(tzinfo=None):
                    # session expired, delete it
                    await self._delete_session(session_id)
                    return None
                
                return response.data["session_data"]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ error getting session: {e}")
            return None
    
    async def _delete_session(self, session_id: str):
        """delete expired session"""
        try:
            if not self.client:
                return
            
            self.client.table("sessions")\
                .delete()\
                .eq("session_id", session_id)\
                .execute()
            
            logger.debug(f"🗑️ deleted expired session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ error deleting session: {e}")
    
    async def cleanup_expired_sessions(self, expire_before: datetime):
        """cleanup expired sessions"""
        try:
            if not self.client or not self.is_initialized:
                return
            
            # delete expired sessions
            expired_sessions = self.client.table("sessions")\
                .delete()\
                .lt("expires_at", expire_before.isoformat())\
                .execute()
            
            # cleanup old conversations (older than 30 days)
            cleanup_before = datetime.utcnow() - timedelta(days=30)
            old_conversations = self.client.table("conversations")\
                .delete()\
                .lt("created_at", cleanup_before.isoformat())\
                .execute()
            
            logger.info("🧹 cleaned up expired sessions and old conversations")
            
        except Exception as e:
            logger.error(f"❌ error cleaning up: {e}")
    
    async def get_analytics_data(self) -> Dict[str, Any]:
        """get analytics data dari supabase"""
        try:
            if not self.client or not self.is_initialized:
                return {}
            
            # total conversations
            total_convs = self.client.table("conversations").select("id", count="exact").execute()
            
            # active sessions
            now = datetime.utcnow().isoformat()
            active_sessions = self.client.table("sessions")\
                .select("id", count="exact")\
                .gt("expires_at", now)\
                .execute()
            
            # recent conversations (last 7 days)
            week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
            recent_convs = self.client.table("conversations")\
                .select("id", count="exact")\
                .gte("created_at", week_ago)\
                .execute()
            
            return {
                "total_conversations": total_convs.count if total_convs else 0,
                "active_sessions": active_sessions.count if active_sessions else 0,
                "recent_conversations": recent_convs.count if recent_convs else 0,
                "message_types": []
            }
            
        except Exception as e:
            logger.error(f"❌ error getting analytics: {e}")
            return {}
    
    async def cleanup(self):
        """cleanup resources"""
        try:
            self.is_initialized = False
            self.client = None
            logger.info("🧹 supabase storage cleaned up")
            
        except Exception as e:
            logger.error(f"❌ error cleaning up: {e}")
    
    async def search_conversations(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """search conversations berdasarkan query"""
        try:
            if not self.client or not self.is_initialized:
                return []
            
            # search dalam question dan response
            response = self.client.table("conversations")\
                .select("*")\
                .or_(f"question.ilike.%{query}%,response.ilike.%{query}%")\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"❌ error searching conversations: {e}")
            return []