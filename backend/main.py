import os
import sys
import json
import logging
import time
import traceback
from typing import List, Dict, Any, Optional
from collections import defaultdict

# setup robust logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# railway deployment info
logger.info("🚀 Starting AI Portfolio Backend...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Platform: {sys.platform}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Files in directory: {sorted(os.listdir('.'))}")

# critical imports with error handling
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import requests
    import re
    import uuid
    logger.info("✅ Core dependencies imported successfully")
except ImportError as e:
    logger.error(f"❌ Critical import failed: {e}")
    sys.exit(1)

# optional imports with fallbacks
DOTENV_AVAILABLE = False
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
    logger.info("✅ python-dotenv loaded")
except ImportError:
    logger.warning("⚠️ python-dotenv not available, using environment variables directly")

# advanced rag imports with fallbacks
RAG_SYSTEM_TYPE = "Built-in Simple RAG"
RAGSystem = None

# try to import advanced rag systems
try:
    from rag_system import RAGSystem, initialize_rag_system
    RAG_SYSTEM_TYPE = "ChromaDB RAG"
    logger.info("✅ ChromaDB RAG system available")
except ImportError:
    try:
        from simple_rag_system import SimpleRAGSystem as RAGSystem, initialize_rag_system
        RAG_SYSTEM_TYPE = "Enhanced Simple RAG"
        logger.info("✅ Enhanced Simple RAG system available")
    except ImportError:
        logger.warning("⚠️ No external RAG systems available, using built-in fallback")

# built-in simple rag fallback
class BuiltinSimpleRAG:
    """ultra-lightweight rag system with zero external dependencies"""
    
    def __init__(self):
        self.knowledge = []
        self.word_index = defaultdict(list)
        
    def load_knowledge_base(self, data: List[Dict]) -> bool:
        try:
            self.knowledge = data
            # build simple word index
            for i, doc in enumerate(data):
                content = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
                keywords = [kw.lower() for kw in doc.get('keywords', [])]
                
                words = re.findall(r'\b\w+\b', content) + keywords
                for word in set(words):
                    if len(word) > 2:
                        self.word_index[word].append(i)
            
            logger.info(f"✅ Built-in RAG loaded {len(data)} documents")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge base: {e}")
            return False
    
    def build_rag_context(self, query: str, top_k: int = 2, category_filter: str = None) -> str:
        if not self.knowledge:
            return ""
        
        query_words = re.findall(r'\b\w+\b', query.lower())
        doc_scores = defaultdict(int)
        
        for word in query_words:
            if word in self.word_index:
                for doc_idx in self.word_index[word]:
                    if category_filter:
                        if self.knowledge[doc_idx].get('category') != category_filter:
                            continue
                    doc_scores[doc_idx] += 1
        
        if not doc_scores:
            return ""
        
        # get top scoring documents
        top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        context_parts = []
        for doc_idx, score in top_docs:
            doc = self.knowledge[doc_idx]
            content = doc.get('content', '')
            if len(content) > 200:
                content = content[:200] + "..."
            context_parts.append(content)
        
        return " ".join(context_parts)
    
    def suggest_related_topics(self, query: str) -> List[str]:
        return ["Keahlian Python", "Web Development Projects", "Data Science Journey"]

# fastapi app initialization
app = FastAPI(
    title="AI Portfolio Backend",
    description="Railway-deployed AI assistant for Danendra's portfolio",
    version="1.0.0"
)

# cors configuration with comprehensive origins
frontend_origins = [
    "https://frontend-portofolio-danen.vercel.app",
    "https://frontend-portofolio-danen-git-master-danenftyessirs-projects.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

# add environment-specific origins
env_frontend = os.getenv("FRONTEND_URL", "")
if env_frontend:
    for url in env_frontend.split(','):
        url = url.strip()
        if url and url not in frontend_origins:
            frontend_origins.append(url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# pydantic models
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class AIResponse(BaseModel):
    response: str
    session_id: str
    related_topics: Optional[List[str]] = []

# lightweight conversation context
class ConversationContext:
    def __init__(self):
        self.last_category: Optional[str] = None
        self.questions_history: List[str] = []
        self.last_updated: float = time.time()
        self.conversation_count: int = 0
    
    def update(self, category: str, question: str, response: str = None):
        self.last_category = category
        self.questions_history.append(question)
        self.conversation_count += 1
        self.last_updated = time.time()
        
        # keep history manageable
        if len(self.questions_history) > 10:
            self.questions_history = self.questions_history[-5:]

# global variables
conversation_sessions = {}
rag_system = None
knowledge_base_data = []

def load_portfolio_knowledge() -> List[Dict]:
    """load portfolio knowledge with comprehensive fallbacks"""
    
    # try to load from portfolio.json
    try:
        if os.path.exists("portfolio.json"):
            with open("portfolio.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"✅ Loaded {len(data)} documents from portfolio.json")
                return data
        else:
            logger.warning("⚠️ portfolio.json not found")
    except Exception as e:
        logger.error(f"❌ Error loading portfolio.json: {e}")
    
    # comprehensive fallback knowledge
    logger.info("Using comprehensive fallback knowledge base")
    return [
        {
            "id": "profil_dasar",
            "category": "profil",
            "title": "Profil Danendra Shafi Athallah",
            "content": "Nama saya Danendra Shafi Athallah, mahasiswa Teknik Informatika ITB semester 4 yang berdomisili di Jakarta. Saya passionate di bidang data science dan algoritma dengan pengalaman 2 tahun web development dan 1 tahun fokus data science. Motto hidup saya adalah 'Menuju tak terbatas dan melampauinya'.",
            "keywords": ["danendra", "itb", "teknik informatika", "data science", "jakarta"]
        },
        {
            "id": "keahlian_python",
            "category": "keahlian", 
            "title": "Keahlian Python & Data Science",
            "content": "Python adalah bahasa utama saya untuk analisis data dan machine learning. Saya menguasai pandas untuk data manipulation, scikit-learn untuk machine learning models, matplotlib dan seaborn untuk visualization, dan numpy untuk numerical computing. Pengalaman 1 tahun khusus di data science dengan berbagai project algoritma kompleks.",
            "keywords": ["python", "data science", "pandas", "scikit-learn", "machine learning"]
        },
        {
            "id": "keahlian_web",
            "category": "keahlian",
            "title": "Keahlian Web Development", 
            "content": "Untuk web development, saya experienced dengan Next.js dan React untuk frontend, plus kemampuan integrate dengan backend services. Portfolio ini sendiri dibuat dengan Next.js + TypeScript, Python FastAPI, dan ditenagai OpenAI. Saya juga menguasai Tailwind CSS untuk styling dan telah mengembangkan berbagai aplikasi web selama 2 tahun terakhir.",
            "keywords": ["nextjs", "react", "web development", "typescript", "tailwind"]
        },
        {
            "id": "proyek_rushhour",
            "category": "proyek",
            "title": "Rush Hour Puzzle Solver",
            "content": "Rush Hour Puzzle Solver adalah project yang paling technically challenging dan educational. Saya implement multiple pathfinding algorithms - UCS untuk optimal solutions, Greedy Best-First untuk speed, A* untuk balanced approach, dan Dijkstra untuk comprehensive exploration. Biggest challenge adalah optimizing algorithm performance untuk handle complex puzzle configurations.",
            "keywords": ["rush hour", "puzzle", "algoritma", "pathfinding", "a*", "dijkstra"]
        },
        {
            "id": "proyek_alchemy",
            "category": "proyek",
            "title": "Little Alchemy Search Algorithm",
            "content": "Little Alchemy Solver project yang paling intellectually stimulating karena involve complex graph theory applications. Implementation cover BFS untuk breadth exploration, DFS untuk depth analysis, dan Bidirectional Search untuk optimal pathfinding dalam recipe combination space.",
            "keywords": ["little alchemy", "algoritma pencarian", "bfs", "dfs", "bidirectional search"]
        },
        {
            "id": "musik_favorit",
            "category": "musik",
            "title": "Selera Musik Oldies & Nostalgia",
            "content": "Selera musik saya nostalgic & oldies. Lagi relate banget sama 'Without You' Air Supply dan 'Sekali Ini Saja' Glenn Fredly. Oldies punya emotional depth dan musical complexity yang susah dicari di modern music. Untuk coding biasanya pakai lo-fi beats atau soundtrack film.",
            "keywords": ["musik", "oldies", "air supply", "glenn fredly", "without you", "lo-fi"]
        },
        {
            "id": "hobi_kuliner",
            "category": "hobi",
            "title": "Penggemar Street Food",
            "content": "Street food Indonesian jadi obsesi saya. Martabak manis, sate ayam, gorengan, ketoprak, batagor - semuanya punya cerita unik. Yang bikin saya obsessed sama street food adalah culture dan experience-nya. Jakarta punya spot-spot legendary kayak Sabang, Pecenongan yang classic banget.",
            "keywords": ["street food", "martabak", "sate", "gorengan", "ketoprak", "jakarta"]
        },
        {
            "id": "rekrutmen_value",
            "category": "rekrutmen",
            "title": "Why You Should Recruit Me", 
            "content": "Saya bawa kombinasi teori dan praktek di data science yang solid. Track record di berbagai algorithmic projects, academic performance yang konsisten, dan keterlibatan aktif di tech community. Yang bikin saya beda adalah learning agility yang tinggi, systematic approach dalam problem decomposition, dan kemampuan communicate technical concepts ke diverse audiences.",
            "keywords": ["rekrutmen", "hire", "data science", "learning agility", "tech community"]
        }
    ]

def categorize_question(question: str) -> str:
    """categorize questions with comprehensive personal content filtering"""
    question_lower = question.lower()
    
    # personal sensitive content detection
    personal_keywords = [
        "pacar", "jodoh", "pacaran", "girlfriend", "boyfriend", "relationship",
        "gaji", "salary", "penghasilan", "uang", "money", "bayaran",
        "alamat", "rumah", "tinggal dimana", "address", "kontak", "nomor",
        "umur", "usia", "age", "tanggal lahir", "birthday",
        "keluarga", "orangtua", "family", "ayah", "ibu", "parents",
        "agama", "religion", "kepercayaan", "tuhan", "pray",
        "kesehatan", "sakit", "health", "penyakit", "dokter"
    ]
    
    if any(keyword in question_lower for keyword in personal_keywords):
        return "personal_sensitive"
    
    # technical categories
    if any(keyword in question_lower for keyword in ["python", "data science", "machine learning", "keahlian", "skill"]):
        return "keahlian"
    elif any(keyword in question_lower for keyword in ["proyek", "project", "rush hour", "little alchemy"]):
        return "proyek"
    elif any(keyword in question_lower for keyword in ["lagu", "musik", "music"]):
        return "musik"
    elif any(keyword in question_lower for keyword in ["makanan", "street food", "kuliner", "food"]):
        return "makanan"
    elif any(keyword in question_lower for keyword in ["rekrut", "hire", "kenapa harus", "why should"]):
        return "rekrutmen"
    else:
        return "general"

def generate_response(question: str, category: str, rag_context: str = None) -> str:
    """generate intelligent responses with comprehensive fallbacks"""
    
    # handle sensitive personal content
    if category == "personal_sensitive":
        personal_redirects = [
            "Info personal prefer tidak dibahas ya. Mending kita discuss technical skills atau project portfolio aku?",
            "Untuk hal personal kurang nyaman share. Kalau tertarik sama technical journey aku, bisa tanya tentang coding experience atau algorithm projects.",
            "Itu agak personal sih. Lebih seru discuss keahlian aku di Python dan web development. Mau tahu tentang yang mana?"
        ]
        import random
        return random.choice(personal_redirects)
    
    # use rag context if available
    if rag_context:
        return f"Berdasarkan yang aku ketahui, {rag_context[:300]}... Mau tahu lebih detail tentang aspek mana?"
    
    # comprehensive category-based responses
    responses = {
        "keahlian": "Aku punya keahlian utama di Python untuk data science, Next.js untuk web development, dan algoritma untuk problem solving. Pengalaman 2 tahun web dev dan 1 tahun fokus data science. Mau tahu lebih detail tentang yang mana?",
        "proyek": "Project unggulan aku adalah Rush Hour Solver dengan multiple pathfinding algorithms (A*, Dijkstra, UCS) dan Little Alchemy Solver dengan graph search techniques. Both projects involve complex algorithm optimization. Ada yang menarik?",
        "musik": "Musik favorit aku oldies kayak Glenn Fredly ('Sekali Ini Saja') dan Air Supply ('Without You'). Punya emotional depth yang susah dicari di modern music. Untuk coding biasanya pakai lo-fi beats atau film soundtracks.",
        "makanan": "Street food Indonesian jadi obsesi aku! Martabak manis, sate ayam, gorengan, ketoprak, batagor - semuanya punya story unik. Jakarta punya spot legendary kayak Sabang dan Pecenongan yang classic banget.",
        "rekrutmen": "Aku bawa kombinasi teori dan praktek di data science yang solid. Track record di algorithmic projects, academic performance konsisten, dan learning agility tinggi. Plus experience sebagai asisten praktikum dan aktif di tech community!",
        "general": "Aku Danendra, mahasiswa Teknik Informatika ITB semester 4 yang passionate di data science dan web development. Punya pengalaman 2 tahun web dev dan 1 tahun data science. Ada yang mau ditanyakan khusus?"
    }
    
    return responses.get(category, "Interesting question! Bisa tanya lebih spesifik tentang technical skills, projects, atau passion aku di tech?")

def call_openai_api(prompt: str) -> str:
    """call openai api with comprehensive error handling"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("⚠️ OpenAI API key not found, using fallback responses")
            raise ValueError("OpenAI API key not found")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "Kamu adalah asisten virtual Danendra yang informatif dan santai. Jawab dengan bahasa Indonesia yang natural, max 4 kalimat."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"OpenAI API error: {response.status_code}")
            raise ValueError(f"OpenAI API error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
        raise

@app.on_event("startup") 
async def startup_event():
    """comprehensive startup with error handling"""
    global rag_system, knowledge_base_data
    
    try:
        logger.info("🚀 Initializing AI Portfolio Backend...")
        
        # load knowledge base
        knowledge_base_data = load_portfolio_knowledge()
        logger.info(f"📚 Knowledge base loaded: {len(knowledge_base_data)} documents")
        
        # initialize rag system with fallbacks
        if RAGSystem and hasattr(RAGSystem, '__name__'):
            logger.info(f"Attempting to initialize {RAG_SYSTEM_TYPE}...")
            try:
                if 'initialize_rag_system' in globals():
                    rag_system = initialize_rag_system(knowledge_base_data, use_openai=False)
                else:
                    rag_system = RAGSystem()
                    rag_system.load_knowledge_base(knowledge_base_data)
                
                if rag_system:
                    logger.info(f"✅ {RAG_SYSTEM_TYPE} initialized successfully")
                else:
                    raise Exception("RAG system initialization returned None")
                    
            except Exception as e:
                logger.warning(f"⚠️ {RAG_SYSTEM_TYPE} failed to initialize: {e}")
                rag_system = None
        
        # fallback to built-in rag if needed
        if not rag_system:
            logger.info("🔄 Initializing built-in Simple RAG as fallback...")
            rag_system = BuiltinSimpleRAG()
            if rag_system.load_knowledge_base(knowledge_base_data):
                logger.info("✅ Built-in Simple RAG initialized successfully")
            else:
                logger.error("❌ All RAG systems failed to initialize")
                rag_system = None
        
        # environment checks
        logger.info("🔍 Environment checks:")
        logger.info(f"   OpenAI API Key: {'✅ SET' if os.getenv('OPENAI_API_KEY') else '❌ NOT SET'}")
        logger.info(f"   Frontend URL: {os.getenv('FRONTEND_URL', 'NOT SET')}")
        logger.info(f"   Port: {os.getenv('PORT', 'NOT SET')}")
        
        logger.info("🎉 Backend startup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # continue startup even with errors

@app.get("/")
async def root():
    """comprehensive health check endpoint"""
    return {
        "message": "✅ AI Portfolio Backend is running successfully!",
        "status": "healthy",
        "deployment": "Railway Production",
        "rag_system": RAG_SYSTEM_TYPE,
        "knowledge_documents": len(knowledge_base_data),
        "rag_available": rag_system is not None,
        "openai_available": bool(os.getenv("OPENAI_API_KEY")),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "working_directory": os.getcwd()
        },
        "endpoints": ["/ask", "/ask-mock", "/rag-status", "/health"]
    }

@app.get("/health")
async def health_check():
    """railway health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    """main ai endpoint with comprehensive error handling"""
    try:
        # input validation
        if len(request.question.strip()) < 2:
            return AIResponse(
                response="Pertanyaanmu terlalu singkat. Coba tanyakan lebih detail tentang diriku.",
                session_id=request.session_id or str(uuid.uuid4())
            )
        
        session_id = request.session_id or str(uuid.uuid4())
        
        # session management
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = ConversationContext()
        
        context = conversation_sessions[session_id]
        category = categorize_question(request.question)
        
        logger.info(f"Question: {request.question[:50]}... | Category: {category}")
        
        # rag context retrieval
        rag_context = ""
        related_topics = []
        
        if rag_system:
            try:
                rag_context = rag_system.build_rag_context(request.question, top_k=2)
                if hasattr(rag_system, 'suggest_related_topics'):
                    related_topics = rag_system.suggest_related_topics(request.question)
            except Exception as e:
                logger.error(f"RAG error: {e}")
        
        # response generation
        if rag_context and os.getenv("OPENAI_API_KEY"):
            try:
                prompt = f"""
Kamu adalah asisten pribadi Danendra yang informatif dan santai.
Jawab dengan bahasa Indonesia yang natural.

Informasi dari knowledge base:
{rag_context}

Pertanyaan: {request.question}

Jawab dengan santai dan to the point, acknowledge pertanyaan, max 4 kalimat.
"""
                response_text = call_openai_api(prompt)
                logger.info("✅ OpenAI response generated")
                
            except Exception as e:
                logger.warning(f"OpenAI failed, using fallback: {e}")
                response_text = generate_response(request.question, category, rag_context)
        else:
            response_text = generate_response(request.question, category, rag_context)
        
        # update conversation context
        context.update(category, request.question, response_text)
        
        return AIResponse(
            response=response_text,
            session_id=session_id,
            related_topics=related_topics[:3] if related_topics else []
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return AIResponse(
            response="Maaf, ada error sistem. Coba lagi dalam beberapa saat ya!",
            session_id=request.session_id or str(uuid.uuid4()),
            related_topics=[]
        )

@app.post("/ask-mock", response_model=AIResponse)
async def ask_ai_mock(request: QuestionRequest):
    """mock endpoint for testing"""
    session_id = request.session_id or str(uuid.uuid4())
    category = categorize_question(request.question)
    
    mock_responses = {
        "keahlian": "Aku punya keahlian di Python untuk data science, Next.js untuk web development, dan algoritma untuk problem solving. Mau tahu lebih detail tentang yang mana?",
        "proyek": "Project favorit aku Rush Hour Solver dengan pathfinding algorithms dan Little Alchemy Search dengan graph theory. Which one interests you?",
        "general": "Hi! Aku Danendra, mahasiswa Teknik Informatika ITB yang passionate di data science dan web development. Ada yang mau ditanyakan?"
    }
    
    response_text = mock_responses.get(category, "Interesting question! Mau tahu tentang technical skills atau project portfolio aku?")
    
    return AIResponse(
        response=response_text,
        session_id=session_id,
        related_topics=["Keahlian Python", "Web Development", "Algorithm Projects"]
    )

@app.get("/rag-status")
async def get_rag_status():
    """comprehensive rag system status"""
    global rag_system, knowledge_base_data
    
    if rag_system:
        return {
            "status": "healthy",
            "system_type": RAG_SYSTEM_TYPE,
            "documents_indexed": len(knowledge_base_data),
            "openai_available": bool(os.getenv("OPENAI_API_KEY")),
            "fallback_mode": False,
            "test_query": "✅ RAG system operational"
        }
    else:
        return {
            "status": "fallback_mode",
            "system_type": "Static Responses",
            "documents_indexed": len(knowledge_base_data),
            "openai_available": bool(os.getenv("OPENAI_API_KEY")), 
            "fallback_mode": True,
            "reason": "RAG system not available, using static responses"
        }

# railway production server configuration
if __name__ == "__main__":
    try:
        import uvicorn
        
        # robust port handling
        port = os.getenv("PORT")
        if not port:
            logger.error("❌ PORT environment variable not found!")
            port = "8000"
            logger.warning(f"🔄 Using default port: {port}")
        
        try:
            port = int(port)
        except ValueError:
            logger.error(f"❌ Invalid PORT value: {port}")
            port = 8000
        
        host = "0.0.0.0"
        
        logger.info(f"🚀 Starting production server on {host}:{port}")
        logger.info(f"Environment: Railway Production")
        logger.info(f"RAG System: {RAG_SYSTEM_TYPE}")
        logger.info(f"Knowledge Base: {len(knowledge_base_data)} documents")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            timeout_keep_alive=120
        )
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)