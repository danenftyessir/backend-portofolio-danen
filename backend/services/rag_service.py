import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

from ..config import Settings
from ..models import PortfolioDocument
from ..storage.supabase_storage import SupabaseStorage
from ..storage.memory_storage import MemoryStorage
from ..rag_system import SimpleRAGSystem, load_knowledge_from_file, initialize_rag_system

logger = logging.getLogger(__name__)

class RAGService:
    """rag service dengan simple text matching dan supabase storage"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.rag_system: Optional[SimpleRAGSystem] = None
        self.documents: List[PortfolioDocument] = []
        self.storage = None
        self.storage_type = "unknown"
        self.is_initialized = False
        
    async def initialize(self):
        """initialize rag service dengan supabase priority"""
        try:
            logger.info("🚀 initializing rag service with supabase...")
            
            # selalu coba supabase dulu sesuai permintaan user
            await self._initialize_storage_with_supabase_priority()
            
            # load portfolio data
            knowledge_data = self._load_portfolio_data()
            
            # initialize simple rag system
            self.rag_system = initialize_rag_system(knowledge_data, use_openai=False)
            
            if self.rag_system:
                self.is_initialized = True
                logger.info(f"✅ rag service initialized successfully with {self.storage_type} storage")
                logger.info(f"📊 loaded {len(self.rag_system.documents)} knowledge documents")
            else:
                logger.error("❌ failed to initialize rag system")
                
        except Exception as e:
            logger.error(f"❌ failed to initialize rag service: {e}")
            raise
    
    async def _initialize_storage_with_supabase_priority(self):
        """initialize storage dengan supabase sebagai prioritas"""
        
        # cek apakah supabase dikonfigurasi
        has_supabase_config = (
            self.settings.supabase_url and 
            self.settings.supabase_key and
            self.settings.supabase_url != "None" and
            self.settings.supabase_key != "None"
        )
        
        if has_supabase_config:
            logger.info("🔗 attempting supabase storage initialization...")
            try:
                self.storage = SupabaseStorage(self.settings)
                await self.storage.initialize()
                self.storage_type = "supabase"
                logger.info("✅ supabase storage initialized successfully!")
                return
            except Exception as e:
                logger.warning(f"⚠️ supabase storage failed: {e}")
                logger.info("📋 supabase troubleshooting:")
                logger.info("   1. check supabase project is active")
                logger.info("   2. verify url and api key are correct")
                logger.info("   3. run table creation sql script")
                logger.info("   4. check network connectivity")
                
                # user minta pakai supabase apapun yang terjadi, tapi kita tetap fallback untuk tidak crash
                logger.info("🔄 falling back to memory storage to keep system running...")
        else:
            logger.info("⚠️ supabase not configured (missing url/key)")
        
        # fallback ke memory storage
        logger.info("🔄 initializing memory storage as fallback...")
        try:
            self.storage = MemoryStorage()
            await self.storage.initialize()
            self.storage_type = "memory"
            logger.info("✅ memory storage initialized (fallback)")
        except Exception as e:
            logger.error(f"❌ even memory storage failed: {e}")
            raise
    
    def _load_portfolio_data(self) -> List[Dict]:
        """load portfolio data dari json file dengan multi-path search"""
        try:
            # semua kemungkinan lokasi file portfolio.json
            search_paths = [
                "data/portfolio.json",                                    # relative dari current working dir
                "backend/data/portfolio.json",                            # jika run dari root
                os.path.join("data", "portfolio.json"),                   # explicit join
                os.path.join("backend", "data", "portfolio.json"),        # explicit join dari root
                os.path.join(os.getcwd(), "data", "portfolio.json"),      # dari current working dir
                os.path.join(os.getcwd(), "backend", "data", "portfolio.json"),  # dari cwd+backend
                os.path.join(os.path.dirname(__file__), "..", "data", "portfolio.json"),  # relative ke file ini
                self.settings.portfolio_data_path                         # dari config
            ]
            
            # coba setiap path
            for path in search_paths:
                if os.path.exists(path):
                    logger.info(f"📂 found portfolio data at: {path}")
                    data = load_knowledge_from_file(path)
                    if data:
                        logger.info(f"📊 loaded {len(data)} documents from portfolio file")
                        return data
                    else:
                        logger.warning(f"⚠️ portfolio file exists but empty: {path}")
            
            # jika tidak ditemukan, debug directory structure
            logger.warning("📂 portfolio.json not found, debugging directory structure:")
            current_dir = os.getcwd()
            logger.warning(f"   current working directory: {current_dir}")
            
            # list files in current dir
            if os.path.exists(current_dir):
                files = [f for f in os.listdir(current_dir) if not f.startswith('.')][:10]
                logger.warning(f"   current dir files: {files}")
            
            # check data directory
            data_paths = ["data", "backend/data", "./data", "./backend/data"]
            for data_path in data_paths:
                if os.path.exists(data_path):
                    files = os.listdir(data_path)
                    logger.warning(f"   {data_path}/ contents: {files}")
            
            logger.warning("🔄 using fallback data due to missing portfolio.json")
            return self._get_fallback_data()
            
        except Exception as e:
            logger.error(f"❌ error loading portfolio data: {e}")
            return self._get_fallback_data()
    
    def _get_fallback_data(self) -> List[Dict]:
        """comprehensive fallback data untuk development"""
        return [
            {
                "id": "profil_danendra",
                "category": "profil",
                "title": "profil danendra shafi athallah",
                "content": "nama saya danendra shafi athallah, mahasiswa teknik informatika itb semester 4 dari jakarta. passionate di bidang data science dan algoritma dengan pengalaman 2 tahun web development. saya active sebagai asisten praktikum dan involved di tech community.",
                "keywords": ["danendra", "profil", "itb", "jakarta", "data science", "algoritma"]
            },
            {
                "id": "keahlian_teknis",
                "category": "keahlian",
                "title": "keahlian programming dan teknologi",
                "content": "keahlian utama saya meliputi python untuk data science dan machine learning, java untuk algorithmic programming, next.js dan react untuk web development, plus various tools seperti git, jupyter notebook, dan figma untuk design.",
                "keywords": ["python", "java", "next.js", "react", "data science", "machine learning", "web development"]
            },
            {
                "id": "proyek_highlight",
                "category": "proyek",
                "title": "project highlights dan achievements",
                "content": "project notable termasuk rush hour puzzle solver dengan multiple pathfinding algorithms (a*, dijkstra, ucs), little alchemy search algorithm dengan graph theory, dan iq puzzler pro solver menggunakan backtracking. juga active di datathon ui dan various coding competitions.",
                "keywords": ["rush hour", "puzzle solver", "algoritma", "pathfinding", "little alchemy", "datathon", "competition"]
            },
            {
                "id": "personal_interests",
                "category": "personal",
                "title": "hobi dan interests personal",
                "content": "saya passionate reader dengan preference ke fantasy novels seperti omniscient reader viewpoint. street food enthusiast yang suka explore kuliner jakarta. music taste cenderung oldies seperti air supply dan glenn fredly. untuk relaxation biasanya nonton film atau drama korea.",
                "keywords": ["membaca", "novel", "fantasy", "street food", "kuliner", "musik", "oldies", "film", "drama korea"]
            }
        ]
    
    async def retrieve_context(self, query: str, max_docs: int = None) -> str:
        """retrieve relevant context untuk query"""
        try:
            if not self.is_initialized or not self.rag_system:
                logger.warning("⚠️ rag system not initialized for context retrieval")
                return ""
            
            max_docs = max_docs or self.settings.max_retrieved_docs
            context = self.rag_system.build_rag_context(query, max_docs)
            
            if context:
                logger.debug(f"🔍 retrieved context for query: {query[:50]}...")
            else:
                logger.debug(f"🔍 no relevant context found for: {query[:50]}...")
            
            return context
            
        except Exception as e:
            logger.error(f"❌ error retrieving context: {e}")
            return ""
    
    async def get_related_topics(self, query: str) -> List[str]:
        """get related topics untuk query"""
        try:
            if not self.is_initialized or not self.rag_system:
                return []
            
            topics = self.rag_system.suggest_related_topics(query)
            logger.debug(f"🏷️ found {len(topics)} related topics for query")
            return topics
            
        except Exception as e:
            logger.error(f"❌ error getting related topics: {e}")
            return []
    
    async def get_status(self) -> Dict[str, Any]:
        """get comprehensive rag service status"""
        try:
            return {
                "status": "healthy" if self.is_initialized else "initializing",
                "storage_type": self.storage_type,
                "vector_store_size": len(self.rag_system.documents) if self.rag_system else 0,
                "last_updated": datetime.utcnow(),
                "embedding_model": "simple_text_matching",
                "supabase_configured": bool(
                    self.settings.supabase_url and self.settings.supabase_key
                ),
                "documents_loaded": len(self.rag_system.documents) if self.rag_system else 0,
                "fallback_data_used": self.storage_type == "memory" and len(self.rag_system.documents) <= 4
            }
        except Exception as e:
            logger.error(f"❌ error getting rag status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def rebuild_index(self):
        """rebuild rag index dengan fresh data"""
        try:
            logger.info("🔄 rebuilding rag index...")
            
            # reload portfolio data
            knowledge_data = self._load_portfolio_data()
            
            # rebuild rag system
            self.rag_system = initialize_rag_system(knowledge_data, use_openai=False)
            
            if self.rag_system:
                logger.info(f"✅ rag index rebuilt with {len(self.rag_system.documents)} documents")
            else:
                raise Exception("failed to rebuild rag system")
                
        except Exception as e:
            logger.error(f"❌ error rebuilding rag index: {e}")
            raise
    
    async def search_documents(self, query: str, category: Optional[str] = None) -> List[Dict]:
        """search documents dengan optional category filter"""
        try:
            if not self.is_initialized or not self.rag_system:
                return []
            
            docs = self.rag_system.retrieve_relevant_docs(
                query, 
                top_k=5, 
                category_filter=category
            )
            
            results = []
            for doc in docs:
                results.append({
                    "title": doc["metadata"]["title"],
                    "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                    "category": doc["metadata"]["category"],
                    "keywords": doc["metadata"].get("keywords", []),
                    "similarity_score": doc["similarity_score"]
                })
            
            logger.debug(f"🔍 search returned {len(results)} documents for query: {query[:50]}")
            return results
            
        except Exception as e:
            logger.error(f"❌ error searching documents: {e}")
            return []
    
    async def cleanup(self):
        """cleanup resources"""
        try:
            if self.storage:
                await self.storage.cleanup()
            
            self.is_initialized = False
            logger.info("🧹 rag service cleaned up")
            
        except Exception as e:
            logger.error(f"❌ error cleaning up rag service: {e}")