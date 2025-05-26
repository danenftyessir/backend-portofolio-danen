from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import logging
import json
import requests
import re
import uuid
from typing import List, Dict, Any, Optional
import time
from collections import defaultdict

# setup logging yang robust untuk Railway
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load environment variables
load_dotenv()

# simple rag system fallback
class SimpleRAG:
    def __init__(self):
        self.knowledge = []
    
    def load_knowledge(self, data):
        self.knowledge = data
        return True
    
    def build_rag_context(self, query, top_k=2, category_filter=None):
        if not self.knowledge:
            return ""
        
        query_lower = query.lower()
        relevant_docs = []
        
        for doc in self.knowledge:
            content = doc.get('content', '').lower()
            title = doc.get('title', '').lower()
            keywords = [kw.lower() for kw in doc.get('keywords', [])]
            
            # simple keyword matching
            score = 0
            for word in query_lower.split():
                if word in content:
                    score += 1
                if word in title:
                    score += 2
                if word in keywords:
                    score += 3
            
            if score > 0:
                relevant_docs.append((doc, score))
        
        # sort by score dan return top docs
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = relevant_docs[:top_k]
        
        if not top_docs:
            return ""
        
        context_parts = []
        for doc, score in top_docs:
            content = doc.get('content', '')
            if len(content) > 200:
                content = content[:200] + "..."
            context_parts.append(content)
        
        return " ".join(context_parts)
    
    def suggest_related_topics(self, query):
        topics = ["Keahlian Python", "Web Development", "Data Science Projects"]
        return topics[:3]

app = FastAPI(title="AI Portfolio Backend")

# cors configuration
frontend_urls = [
    "https://frontend-portofolio-danen.vercel.app",
    "https://frontend-portofolio-danen-git-master-danenftyessirs-projects.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

# environment specific origins
env_frontend = os.getenv("FRONTEND_URL", "")
if env_frontend:
    for url in env_frontend.split(','):
        url = url.strip()
        if url and url not in frontend_urls:
            frontend_urls.append(url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_urls,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class AIResponse(BaseModel):
    response: str
    session_id: str
    related_topics: Optional[List[str]] = []

# simple conversation context
class ConversationContext:
    def __init__(self):
        self.last_category: Optional[str] = None
        self.questions_history: List[str] = []
        self.last_updated: float = time.time()
    
    def update(self, category: str, question: str, response: str = None):
        self.last_category = category
        self.questions_history.append(question)
        self.last_updated = time.time()
        
        # keep history manageable
        if len(self.questions_history) > 10:
            self.questions_history = self.questions_history[-5:]

# global variables
conversation_sessions = {}
rag_system = None
knowledge_base_data = []

def load_portfolio_knowledge() -> List[Dict]:
    """load knowledge dengan robust error handling"""
    try:
        if os.path.exists("portfolio.json"):
            with open("portfolio.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"✅ loaded {len(data)} documents from portfolio.json")
                return data
        else:
            logger.warning("⚠️ portfolio.json not found, using fallback knowledge")
            return get_fallback_knowledge()
    except Exception as e:
        logger.error(f"❌ error loading portfolio.json: {e}")
        return get_fallback_knowledge()

def get_fallback_knowledge() -> List[Dict]:
    """comprehensive fallback knowledge"""
    return [
        {
            "id": "profil_dasar",
            "category": "profil",
            "title": "Profil Danendra",
            "content": "Danendra Shafi Athallah, mahasiswa Teknik Informatika ITB semester 4 yang passionate di data science dan web development. Punya pengalaman 2 tahun web development dan 1 tahun data science.",
            "keywords": ["danendra", "itb", "teknik informatika", "data science", "web development"]
        },
        {
            "id": "keahlian_python",
            "category": "keahlian",
            "title": "Keahlian Python & Data Science",
            "content": "Menguasai Python untuk data science dengan pandas, scikit-learn, machine learning. Juga experienced dengan Next.js dan React untuk web development. Portfolio ini dibuat dengan Next.js + TypeScript dan Python FastAPI.",
            "keywords": ["python", "data science", "pandas", "machine learning", "next.js", "react"]
        },
        {
            "id": "proyek_rushhour",
            "category": "proyek",
            "title": "Rush Hour Puzzle Solver",
            "content": "Project algoritma pathfinding dengan implementasi UCS, A*, Greedy Best-First Search, dan Dijkstra untuk menyelesaikan puzzle Rush Hour. Termasuk visualisasi interaktif dan optimisasi performa.",
            "keywords": ["rush hour", "algoritma", "pathfinding", "a*", "dijkstra", "puzzle"]
        },
        {
            "id": "proyek_alchemy",
            "category": "proyek", 
            "title": "Little Alchemy Search Algorithm",
            "content": "Implementasi BFS, DFS, dan Bidirectional Search untuk mencari kombinasi recipe dalam permainan Little Alchemy 2. Melibatkan graph theory dan optimisasi search strategies.",
            "keywords": ["little alchemy", "bfs", "dfs", "bidirectional search", "graph theory"]
        },
        {
            "id": "hobi_music",
            "category": "musik",
            "title": "Selera Musik & Hobi",
            "content": "Suka musik oldies seperti Glenn Fredly dan Air Supply, terutama 'Without You' dan 'Sekali Ini Saja'. Untuk coding biasanya pakai lo-fi beats. Hobi lain termasuk membaca novel fantasy dan hunting street food Jakarta.",
            "keywords": ["musik", "glenn fredly", "air supply", "oldies", "lo-fi", "street food"]
        },
        {
            "id": "rekrutmen_value",
            "category": "rekrutmen",
            "title": "Why Recruit Me",
            "content": "Kombinasi teori dan praktek data science yang solid, track record akademik yang konsisten, dan learning agility tinggi. Experience sebagai asisten praktikum dan aktif di tech community. Bisa adapt cepat dengan teknologi baru.",
            "keywords": ["rekrutmen", "data science", "learning agility", "asisten praktikum", "tech community"]
        }
    ]

def categorize_question(question: str) -> str:
    """simple but effective question categorization"""
    question_lower = question.lower()
    
    # personal sensitive content
    personal_keywords = ["pacar", "gaji", "alamat", "umur", "keluarga", "agama", "kesehatan"]
    if any(keyword in question_lower for keyword in personal_keywords):
        return "personal_sensitive"
    
    # technical categories
    if any(keyword in question_lower for keyword in ["python", "data science", "machine learning", "keahlian", "skill"]):
        return "keahlian"
    elif any(keyword in question_lower for keyword in ["proyek", "project", "rush hour", "little alchemy"]):
        return "proyek"
    elif any(keyword in question_lower for keyword in ["lagu", "musik"]):
        return "musik"
    elif any(keyword in question_lower for keyword in ["makanan", "street food", "kuliner"]):
        return "makanan"
    elif any(keyword in question_lower for keyword in ["rekrut", "hire", "kenapa harus"]):
        return "rekrutmen"
    else:
        return "general"

def generate_response(question: str, category: str, rag_context: str = None) -> str:
    """generate intelligent response"""
    
    # handle sensitive content
    if category == "personal_sensitive":
        return "Info personal prefer tidak dibahas ya. Mending kita discuss technical skills atau project portfolio aku?"
    
    # use rag context if available
    if rag_context:
        return f"Berdasarkan yang aku ketahui, {rag_context[:250]}... Mau tahu lebih detail tentang aspek mana?"
    
    # category-based responses
    responses = {
        "keahlian": "Aku punya keahlian utama di Python untuk data science, Next.js untuk web development, dan algoritma untuk problem solving. Mau tahu lebih detail tentang yang mana?",
        "proyek": "Project unggulan aku Rush Hour Solver dengan algoritma pathfinding dan Little Alchemy Solver dengan graph search. Ada yang menarik?",
        "musik": "Musik favorit aku oldies kayak Glenn Fredly dan Air Supply. Enak buat coding juga pakai lo-fi beats.",
        "makanan": "Street food Indonesia jadi obsesi aku! Martabak, sate, ketoprak, semuanya enak banget.",
        "rekrutmen": "Aku bawa kombinasi teori dan praktek di data science yang solid. Track record akademik bagus dan passionate banget di tech!",
        "general": "Aku Danendra, mahasiswa Teknik Informatika ITB yang passionate di data science dan web development. Ada yang mau ditanyakan khusus?"
    }
    
    return responses.get(category, "Interesting question! Bisa tanya lebih spesifik tentang technical skills atau project portfolio aku?")

def call_openai_api(prompt: str) -> str:
    """call openai with robust error handling"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "Kamu adalah asisten virtual Danendra yang informatif dan santai. Jawab dengan bahasa Indonesia yang natural."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 600,
            "temperature": 0.8
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise ValueError(f"OpenAI API error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """startup with comprehensive error handling"""
    global rag_system, knowledge_base_data
    
    try:
        logger.info("🚀 Starting AI Portfolio Backend...")        
        # load knowledge base
        knowledge_base_data = load_portfolio_knowledge()
        logger.info(f"📚 Knowledge base: {len(knowledge_base_data)} documents")
        rag_system = SimpleRAG()
        if rag_system.load_knowledge(knowledge_base_data):
            logger.info("✅ Simple RAG system initialized")
        else:
            logger.warning("⚠️ RAG system initialization failed")
        
        logger.info("🎉 Backend startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

@app.get("/")
async def root():
    """health check endpoint"""
    return {
        "message": "AI Portfolio Backend is running!",
        "status": "healthy",
        "knowledge_docs": len(knowledge_base_data),
        "rag_available": rag_system is not None,
        "endpoints": ["/ask", "/ask-mock", "/rag-status"]
    }

@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    """main ai endpoint"""
    try:
        if len(request.question.strip()) < 2:
            return AIResponse(
                response="Pertanyaanmu terlalu singkat. Coba tanyakan lebih detail tentang diriku.",
                session_id=request.session_id or str(uuid.uuid4())
            )
        
        session_id = request.session_id or str(uuid.uuid4())
        
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = ConversationContext()
        
        context = conversation_sessions[session_id]
        category = categorize_question(request.question)
        
        logger.info(f"Question: {request.question[:50]}... | Category: {category}")
        
        # get rag context
        rag_context = ""
        related_topics = []
        
        if rag_system:
            try:
                rag_context = rag_system.build_rag_context(request.question, top_k=2)
                related_topics = rag_system.suggest_related_topics(request.question)
            except Exception as e:
                logger.error(f"RAG error: {e}")
        
        # try openai if context available
        if rag_context and os.getenv("OPENAI_API_KEY"):
            try:
                prompt = f"""
Kamu adalah asisten pribadi Danen yang informatif dan santai.
Jawab dengan bahasa Indonesia yang natural.

Informasi: {rag_context}

Pertanyaan: {request.question}

Jawab dengan santai dan to the point, max 3-4 kalimat.
"""
                response_text = call_openai_api(prompt)
                logger.info("✅ OpenAI response generated")
                
            except Exception as e:
                logger.warning(f"OpenAI failed, using fallback: {e}")
                response_text = generate_response(request.question, category, rag_context)
        else:
            response_text = generate_response(request.question, category, rag_context)
        
        context.update(category, request.question, response_text)
        
        return AIResponse(
            response=response_text,
            session_id=session_id,
            related_topics=related_topics[:3]
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return AIResponse(
            response="Maaf, ada error sistem. Coba lagi dalam beberapa saat.",
            session_id=request.session_id or str(uuid.uuid4())
        )

@app.post("/ask-mock", response_model=AIResponse)
async def ask_ai_mock(request: QuestionRequest):
    """mock endpoint untuk testing"""
    session_id = request.session_id or str(uuid.uuid4())
    category = categorize_question(request.question)
    
    mock_responses = {
        "keahlian": "Aku punya keahlian di Python, data science, dan web development dengan Next.js. Mau tahu lebih detail?",
        "proyek": "Project favorit aku Rush Hour Solver dan Little Alchemy Search Algorithm. Which one interests you?",
        "general": "Hi! Aku Danendra, mahasiswa Teknik Informatika ITB. Ada yang mau ditanyakan tentang technical journey aku?"
    }
    
    response_text = mock_responses.get(category, "Interesting question! Tell me more about what you'd like to know.")
    
    return AIResponse(
        response=response_text,
        session_id=session_id,
        related_topics=["Keahlian Python", "Project Highlights", "Data Science Journey"]
    )

@app.get("/rag-status")
async def get_rag_status():
    """rag system status"""
    global rag_system, knowledge_base_data
    
    if rag_system:
        return {
            "status": "healthy",
            "system_type": "Simple",
            "documents_indexed": len(knowledge_base_data),
            "openai_available": bool(os.getenv("OPENAI_API_KEY")),
            "fallback_mode": False
        }
    else:
        return {
            "status": "fallback_mode",
            "system_type": "Static",
            "documents_indexed": len(knowledge_base_data),
            "openai_available": bool(os.getenv("OPENAI_API_KEY")),
            "fallback_mode": True
        }

# production server configuration untuk Railway
if __name__ == "__main__":
    import uvicorn
    
    # robust port handling untuk Railway
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    try:
        uvicorn.run(
            "main:app", 
            host=host, 
            port=port,
            timeout_keep_alive=120,
            access_log=True
        )
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        raise