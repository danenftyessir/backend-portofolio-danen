import os
import sys
import json
import time
import uuid
import hashlib
import gc
from typing import List, Dict, Any, Optional
from collections import defaultdict

# minimal imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# setup logging
def log(message: str, level: str = "INFO"):
    print(f"[{level}] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

log("🚀 Starting ChromaDB Lightweight Backend")

# chromadb dengan config minimal
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
    
    # custom lightweight embedding function
    class LightweightEmbeddingFunction(EmbeddingFunction):
        """super lightweight embedding using hash-based approach"""
        
        def __init__(self, embedding_dim: int = 128):  # dimensi kecil
            self.embedding_dim = embedding_dim
        
        def __call__(self, input: Documents) -> Embeddings:
            """generate embeddings using lightweight hash approach"""
            embeddings = []
            
            for text in input:
                # preprocess text
                text_lower = text.lower()
                words = text_lower.split()
                
                # create embedding vector
                embedding = [0.0] * self.embedding_dim
                
                # hash-based embedding dengan TF-IDF style weighting
                word_counts = defaultdict(int)
                for word in words:
                    if len(word) > 2:  # skip short words
                        word_counts[word] += 1
                
                # generate embedding
                for word, count in word_counts.items():
                    # use multiple hash functions untuk distribusi yang lebih baik
                    for i in range(3):  # 3 hash functions
                        hash_val = int(hashlib.md5(f"{word}_{i}".encode()).hexdigest(), 16)
                        idx = hash_val % self.embedding_dim
                        
                        # tf-idf style weighting
                        tf = count / len(words) if words else 0
                        embedding[idx] += tf
                
                # normalize embedding
                norm = sum(x*x for x in embedding) ** 0.5
                if norm > 0:
                    embedding = [x/norm for x in embedding]
                
                embeddings.append(embedding)
            
            return embeddings
    
    CHROMADB_AVAILABLE = True
    log("✅ ChromaDB available")
    
except ImportError:
    CHROMADB_AVAILABLE = False
    log("❌ ChromaDB not available")

# fastapi app
app = FastAPI(
    title="AI Portfolio Backend - ChromaDB Lightweight",
    description="Optimized ChromaDB for Render free tier",
    version="3.0.0"
)

# cors
frontend_origins = [
    "https://frontend-portofolio-danen.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

env_frontend = os.getenv("FRONTEND_URL", "")
if env_frontend:
    frontend_origins.append(env_frontend)

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# models
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class AIResponse(BaseModel):
    response: str
    session_id: str
    related_topics: Optional[List[str]] = []

# lightweight chromadb rag system
class ChromaDBLightweightRAG:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_function = LightweightEmbeddingFunction(embedding_dim=128)
        self.is_initialized = False
        
    def initialize(self):
        """lazy initialization untuk hemat memory"""
        if self.is_initialized:
            return True
            
        try:
            # chromadb client dengan config minimal
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",  # lebih ringan dari sqlite
                persist_directory="./chroma_lightweight",
                anonymized_telemetry=False
            ))
            
            # delete collection lama jika ada
            try:
                self.client.delete_collection("portfolio_lightweight")
            except:
                pass
            
            # create collection baru
            self.collection = self.client.create_collection(
                name="portfolio_lightweight",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # gunakan cosine similarity
            )
            
            self.is_initialized = True
            log("✅ ChromaDB initialized")
            return True
            
        except Exception as e:
            log(f"❌ ChromaDB init failed: {e}", "ERROR")
            return False
    
    def add_documents(self, documents: List[Dict]):
        """add documents ke chromadb"""
        if not self.is_initialized:
            if not self.initialize():
                return False
        
        try:
            # prepare data
            texts = []
            metadatas = []
            ids = []
            
            for doc in documents:
                # combine text untuk embedding
                text = f"{doc.get('title', '')} {doc.get('content', '')} {' '.join(doc.get('keywords', []))}"
                texts.append(text)
                
                # metadata
                metadatas.append({
                    "title": doc.get('title', ''),
                    "category": doc.get('category', ''),
                    "id": doc.get('id', '')
                })
                
                ids.append(doc.get('id', str(uuid.uuid4())))
            
            # add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            log(f"✅ Added {len(documents)} documents to ChromaDB")
            
            # force garbage collection
            gc.collect()
            
            return True
            
        except Exception as e:
            log(f"❌ Error adding documents: {e}", "ERROR")
            return False
    
    def query(self, query_text: str, n_results: int = 3, category_filter: str = None):
        """query chromadb"""
        if not self.is_initialized:
            return []
            
        try:
            # prepare where clause
            where = None
            if category_filter:
                where = {"category": category_filter}
            
            # query
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where
            )
            
            # parse results
            if results and results['documents'] and len(results['documents'][0]) > 0:
                docs = []
                for i in range(len(results['documents'][0])):
                    docs.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else 0
                    })
                return docs
            
            return []
            
        except Exception as e:
            log(f"❌ Query error: {e}", "ERROR")
            return []
    
    def cleanup(self):
        """cleanup untuk hemat memory"""
        if self.collection:
            self.collection = None
        if self.client:
            self.client = None
        self.is_initialized = False
        gc.collect()

# global instances
rag_system = None
conversation_sessions = {}
knowledge_base = []

# load knowledge base
def load_portfolio_knowledge():
    """load portfolio knowledge"""
    try:
        if os.path.exists("portfolio.json"):
            with open("portfolio.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                log(f"✅ Loaded {len(data)} documents")
                return data
    except Exception as e:
        log(f"❌ Error loading portfolio.json: {e}", "ERROR")
    
    # fallback data
    return [
        {
            "id": "profil_dasar",
            "category": "profil",
            "title": "Profil Dasar Danendra",
            "content": "Nama saya Danendra Shafi Athallah, mahasiswa Teknik Informatika ITB semester 4 yang berdomisili di Jakarta. Saya passionate di bidang data science dan algoritma.",
            "keywords": ["nama", "profil", "itb", "jakarta"]
        },
        {
            "id": "keahlian_python",
            "category": "keahlian",
            "title": "Keahlian Python",
            "content": "Python adalah bahasa utama saya untuk data science. Menguasai pandas, scikit-learn, matplotlib untuk analisis data dan machine learning.",
            "keywords": ["python", "data science", "machine learning"]
        },
        {
            "id": "proyek_rushhour",
            "category": "proyek",
            "title": "Rush Hour Solver",
            "content": "Rush Hour Solver dengan algoritma pathfinding - UCS, Greedy, A*, dan Dijkstra untuk menyelesaikan puzzle secara optimal.",
            "keywords": ["rush hour", "algoritma", "pathfinding"]
        }
    ]

# categorize question
def categorize_question(question: str) -> str:
    """categorize question"""
    q_lower = question.lower()
    
    if any(k in q_lower for k in ["python", "java", "skill", "keahlian"]):
        return "keahlian"
    elif any(k in q_lower for k in ["proyek", "project", "rush", "alchemy"]):
        return "proyek"
    elif any(k in q_lower for k in ["nama", "siapa", "kuliah", "itb"]):
        return "profil"
    elif any(k in q_lower for k in ["rekrut", "hire", "kenapa"]):
        return "rekrutmen"
    
    return "general"

# generate response
def generate_response(question: str, rag_context: List[Dict]) -> str:
    """generate response dengan rag context"""
    
    # build context string
    context_str = ""
    if rag_context:
        for doc in rag_context[:2]:  # limit context
            content = doc.get('content', '')
            if len(content) > 150:
                content = content[:150] + "..."
            context_str += f"{content}\n"
    
    # try openai
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and context_str:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            prompt = f"""Kamu adalah AI assistant Danendra. Jawab dalam bahasa Indonesia yang santai.

Konteks dari knowledge base:
{context_str}

Pertanyaan: {question}

Jawab singkat dalam 2-3 kalimat."""

            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "Kamu AI assistant portfolio Danendra."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
                
    except Exception as e:
        log(f"OpenAI error: {e}", "ERROR")
    
    # fallback response dengan context
    if context_str:
        return f"Berdasarkan informasi yang ada, {context_str[:100]}..."
    
    # ultimate fallback
    category = categorize_question(question)
    fallbacks = {
        "keahlian": "Aku punya keahlian di Python untuk data science dan web development dengan Next.js.",
        "proyek": "Proyek unggulan aku adalah Rush Hour Solver dan Little Alchemy Solver dengan algoritma kompleks.",
        "profil": "Aku Danendra, mahasiswa Teknik Informatika ITB yang passionate di data science!",
        "general": "Aku Danendra, mahasiswa ITB yang fokus di data science dan algoritma."
    }
    
    return fallbacks.get(category, fallbacks["general"])

# endpoints
@app.get("/")
async def root():
    return {
        "message": "AI Portfolio Backend - ChromaDB Lightweight",
        "status": "healthy",
        "chromadb": CHROMADB_AVAILABLE,
        "memory_optimized": True
    }

@app.post("/ask")
async def ask_ai(request: QuestionRequest):
    """main endpoint"""
    global rag_system
    
    try:
        # lazy init rag system
        if CHROMADB_AVAILABLE and not rag_system:
            rag_system = ChromaDBLightweightRAG()
            if rag_system.initialize():
                rag_system.add_documents(knowledge_base)
        
        # get rag context
        rag_context = []
        related_topics = []
        
        if rag_system:
            category = categorize_question(request.question)
            rag_context = rag_system.query(
                request.question, 
                n_results=3,
                category_filter=category if category != "general" else None
            )
            
            # extract related topics
            for doc in rag_context[:3]:
                title = doc.get('metadata', {}).get('title', '')
                if title:
                    related_topics.append(title)
        
        # generate response
        response_text = generate_response(request.question, rag_context)
        
        # cleanup sessions untuk hemat memory
        if len(conversation_sessions) > 20:
            oldest = min(conversation_sessions.keys())
            del conversation_sessions[oldest]
        
        session_id = request.session_id or str(uuid.uuid4())
        conversation_sessions[session_id] = time.time()
        
        # periodic cleanup
        if len(conversation_sessions) % 10 == 0:
            gc.collect()
        
        return AIResponse(
            response=response_text,
            session_id=session_id,
            related_topics=related_topics[:3]
        )
        
    except Exception as e:
        log(f"Error: {e}", "ERROR")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag-status")
async def rag_status():
    """check rag status"""
    if rag_system and rag_system.is_initialized:
        return {
            "status": "healthy",
            "type": "ChromaDB Lightweight",
            "embedding_dim": 128,
            "documents": len(knowledge_base)
        }
    else:
        return {
            "status": "not_initialized",
            "chromadb_available": CHROMADB_AVAILABLE
        }

@app.on_event("startup")
async def startup():
    """startup event"""
    global knowledge_base
    
    log("🔧 Starting up...")
    knowledge_base = load_portfolio_knowledge()
    
    # tidak init chromadb di startup untuk hemat memory
    # akan di-init lazy saat pertama kali digunakan
    
    log("✅ Ready!")

@app.on_event("shutdown")
async def shutdown():
    """cleanup on shutdown"""
    if rag_system:
        rag_system.cleanup()
    gc.collect()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)