import os
import sys
import json
import logging
import time
import uuid
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# setup logging minimal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("🚀 starting optimized ai portfolio backend for render (512mb)")

# core imports saja
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import requests
    logger.info("✅ core dependencies loaded")
except ImportError as e:
    logger.error(f"❌ critical import failed: {e}")
    sys.exit(1)

# sqlite vector storage
try:
    from sqlite_vector_storage import SQLiteVectorStorage, initialize_sqlite_rag
    logger.info("✅ sqlite vector storage loaded")
except ImportError:
    logger.error("❌ sqlite_vector_storage.py not found")
    sys.exit(1)

# fastapi app
app = FastAPI(
    title="AI Portfolio Backend (Optimized)",
    description="Lightweight RAG system dengan SQLite untuk Render deployment",
    version="2.0.0"
)

# cors configuration
frontend_origins = [
    "https://frontend-portofolio-danen.vercel.app",
    "https://frontend-portofolio-danen-git-master-danenftyessirs-projects.vercel.app",
    "http://localhost:3000"
]

env_frontend = os.getenv("FRONTEND_URL", "")
if env_frontend:
    frontend_origins.extend([url.strip() for url in env_frontend.split(',') if url.strip()])

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

# simple conversation context
class ConversationContext:
    def __init__(self):
        self.history: List[Tuple[str, str]] = []  # (question, response) pairs
        self.last_updated: float = time.time()
    
    def add(self, question: str, response: str):
        self.history.append((question, response))
        if len(self.history) > 5:  # keep only last 5
            self.history.pop(0)
        self.last_updated = time.time()
    
    def get_context(self) -> str:
        if not self.history:
            return ""
        last_q, last_r = self.history[-1]
        return f"Pertanyaan sebelumnya: {last_q[:50]}..."

# global state minimal
conversation_sessions = {}
rag_system: Optional[SQLiteVectorStorage] = None

def load_portfolio_knowledge() -> List[Dict]:
    """load knowledge base dari file atau default"""
    try:
        if os.path.exists("portfolio.json"):
            with open("portfolio.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"✅ loaded {len(data)} documents from portfolio.json")
                return data
    except Exception as e:
        logger.error(f"❌ error loading portfolio.json: {e}")
    
    # default knowledge base minimal
    return [
        {
            "id": "profil",
            "category": "profil",
            "title": "Profil Danendra",
            "content": "Danendra Shafi Athallah, mahasiswa Teknik Informatika ITB semester 4 dari Jakarta. Passionate di data science dan algoritma dengan 2 tahun web development dan 1 tahun data science.",
            "keywords": ["danendra", "itb", "informatika", "jakarta"]
        },
        {
            "id": "keahlian_python",
            "category": "keahlian",
            "title": "Python & Data Science",
            "content": "Menguasai Python untuk data science: pandas, scikit-learn, matplotlib, numpy. Berpengalaman membuat model machine learning dan analisis data.",
            "keywords": ["python", "data science", "pandas", "machine learning"]
        },
        {
            "id": "keahlian_web",
            "category": "keahlian",
            "title": "Web Development",
            "content": "Experienced dengan Next.js, React, TypeScript, Tailwind CSS. Portfolio ini dibuat dengan Next.js + Python FastAPI.",
            "keywords": ["nextjs", "react", "typescript", "web"]
        },
        {
            "id": "proyek_rushhour",
            "category": "proyek",
            "title": "Rush Hour Solver",
            "content": "Implementasi algoritma pathfinding (UCS, Greedy, A*, Dijkstra) untuk menyelesaikan puzzle Rush Hour dengan visualisasi interaktif.",
            "keywords": ["rush hour", "algoritma", "pathfinding", "puzzle"]
        },
        {
            "id": "proyek_alchemy",
            "category": "proyek",
            "title": "Little Alchemy Solver",
            "content": "Aplikasi pencarian resep Little Alchemy menggunakan BFS, DFS, dan Bidirectional Search dengan optimasi graph theory.",
            "keywords": ["little alchemy", "bfs", "dfs", "graph"]
        },
        {
            "id": "rekrutmen",
            "category": "rekrutmen",
            "title": "Why Hire Me",
            "content": "Kombinasi solid teori dan praktek data science, track record project sukses, learning agility tinggi, dan kemampuan komunikasi technical concepts.",
            "keywords": ["rekrutmen", "hire", "kelebihan", "value"]
        }
    ]

def categorize_question(question: str) -> str:
    """simple question categorization"""
    q_lower = question.lower()
    
    # personal protection
    personal_keywords = ["pacar", "umur", "alamat", "agama", "gaji"]
    if any(kw in q_lower for kw in personal_keywords):
        return "personal"
    
    # technical categories
    if any(kw in q_lower for kw in ["python", "java", "react", "skill", "keahlian"]):
        return "keahlian"
    elif any(kw in q_lower for kw in ["proyek", "project", "rush hour", "alchemy"]):
        return "proyek"
    elif any(kw in q_lower for kw in ["rekrut", "hire", "kenapa"]):
        return "rekrutmen"
    
    return "general"

def create_prompt(question: str, rag_context: str, conv_context: str = "") -> str:
    """create prompt untuk openai"""
    prompt = f"""
Kamu adalah asisten AI Danendra Shafi Athallah. Jawab dengan bahasa Indonesia yang santai dan informatif.

INFORMASI DARI DATABASE:
{rag_context}

{f"KONTEKS PERCAKAPAN: {conv_context}" if conv_context else ""}

Pertanyaan: {question}

Instruksi:
- Jawab dengan natural dan to the point
- Gunakan informasi dari database
- Bahasa Indonesia santai
- Max 3-4 kalimat
"""
    return prompt

def generate_fallback_response(question: str, category: str) -> str:
    """fallback responses"""
    if category == "personal":
        return "Info pribadi prefer tidak dibahas ya. Mending tanya tentang skill atau project aku!"
    
    responses = {
        "keahlian": "Aku punya keahlian di Python untuk data science dan Next.js untuk web development.",
        "proyek": "Project favorit aku Rush Hour Solver dan Little Alchemy Solver, keduanya pakai algoritma kompleks.",
        "rekrutmen": "Aku bawa kombinasi solid antara teori dan praktek di data science dengan track record project sukses.",
        "general": "Hai! Aku Danendra, mahasiswa Teknik Informatika ITB yang passionate di data science."
    }
    
    return responses.get(category, "Aku Danendra, passionate di data science dan web development!")

def call_openai_api(prompt: str) -> Optional[str]:
    """call openai api dengan error handling"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "Kamu adalah asisten AI yang membantu menjawab pertanyaan tentang portfolio."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.8
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        
        return None
        
    except Exception as e:
        logger.error(f"openai api error: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """startup initialization"""
    global rag_system
    
    try:
        logger.info("initializing sqlite rag system...")
        
        # load knowledge
        knowledge_data = load_portfolio_knowledge()
        
        # initialize rag
        rag_system = initialize_sqlite_rag(knowledge_data)
        
        if rag_system:
            stats = rag_system.get_stats()
            logger.info(f"✅ rag system ready: {stats}")
        else:
            logger.error("❌ rag system initialization failed")
        
        logger.info("✅ startup completed!")
        
    except Exception as e:
        logger.error(f"❌ startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """cleanup on shutdown"""
    global rag_system
    if rag_system:
        rag_system.close()

@app.get("/")
async def root():
    """health check endpoint"""
    stats = rag_system.get_stats() if rag_system else {}
    
    return {
        "message": "AI Portfolio Backend (Optimized for Render)",
        "status": "healthy",
        "memory_optimized": True,
        "rag_system": "SQLite Vector Storage",
        "rag_stats": stats,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    """main ai endpoint"""
    try:
        # validate input
        if len(request.question.strip()) < 2:
            return AIResponse(
                response="Pertanyaanmu terlalu singkat. Coba tanyakan lebih detail!",
                session_id=request.session_id or str(uuid.uuid4())
            )
        
        # session management
        session_id = request.session_id or str(uuid.uuid4())
        
        # cleanup old sessions (memory optimization)
        current_time = time.time()
        sessions_to_remove = [
            sid for sid, ctx in conversation_sessions.items()
            if current_time - ctx.last_updated > 900  # 15 minutes
        ]
        for sid in sessions_to_remove:
            del conversation_sessions[sid]
        
        # get or create context
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = ConversationContext()
        
        context = conversation_sessions[session_id]
        
        # categorize question
        category = categorize_question(request.question)
        
        # handle personal questions
        if category == "personal":
            response = generate_fallback_response(request.question, category)
            context.add(request.question, response)
            return AIResponse(response=response, session_id=session_id)
        
        # get rag context
        rag_context = ""
        related_topics = []
        
        if rag_system:
            rag_context = rag_system.build_context(request.question)
            related_topics = rag_system.suggest_topics(request.question)
        
        # try openai first
        response_text = None
        if rag_context:
            prompt = create_prompt(
                request.question, 
                rag_context,
                context.get_context()
            )
            response_text = call_openai_api(prompt)
        
        # fallback if openai fails
        if not response_text:
            if rag_context:
                # use rag context for response
                response_text = f"Berdasarkan yang aku tahu, {rag_context[:200]}... Ada yang mau ditanyakan lebih detail?"
            else:
                # pure fallback
                response_text = generate_fallback_response(request.question, category)
        
        # update context
        context.add(request.question, response_text)
        
        return AIResponse(
            response=response_text,
            session_id=session_id,
            related_topics=related_topics
        )
        
    except Exception as e:
        logger.error(f"error in ask endpoint: {e}")
        return AIResponse(
            response="Maaf ada error. Coba lagi ya!",
            session_id=request.session_id or str(uuid.uuid4())
        )

@app.post("/ask-mock", response_model=AIResponse)
async def ask_ai_mock(request: QuestionRequest):
    """mock endpoint for testing"""
    session_id = request.session_id or str(uuid.uuid4())
    
    # get rag context
    rag_context = ""
    related_topics = []
    
    if rag_system:
        rag_context = rag_system.build_context(request.question)
        related_topics = rag_system.suggest_topics(request.question)
    
    if rag_context:
        response = f"[Mock] Berdasarkan database: {rag_context[:150]}..."
    else:
        response = "[Mock] Aku Danendra, mahasiswa ITB yang passionate di data science!"
    
    return AIResponse(
        response=response,
        session_id=session_id,
        related_topics=related_topics
    )

@app.get("/rag-status")
async def get_rag_status():
    """rag system status"""
    if rag_system:
        return {
            "status": "healthy",
            "system": "SQLite Vector Storage",
            "stats": rag_system.get_stats(),
            "memory_efficient": True
        }
    else:
        return {
            "status": "not_initialized",
            "system": "None"
        }

# untuk local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)