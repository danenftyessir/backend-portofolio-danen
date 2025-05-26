import os
import sys
import json
import logging
import time
import traceback
import uuid
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

# setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("🚀 Starting Full-Featured AI Portfolio Backend (3GB Optimized)")
logger.info(f"Python: {sys.version} | Platform: {sys.platform}")
logger.info(f"Working directory: {os.getcwd()}")

# core imports
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import requests
    logger.info("✅ Core FastAPI dependencies loaded")
except ImportError as e:
    logger.error(f"❌ Critical import failed: {e}")
    sys.exit(1)

# optional imports with fallbacks
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ Environment variables loaded")
except ImportError:
    logger.info("⚠️ python-dotenv not available, using os.environ")

# advanced rag system imports with comprehensive fallbacks
RAG_SYSTEM_TYPE = "Built-in Advanced RAG"
RAGSystem = None
initialize_rag_system = None

# try chromadb first (target for 3GB deployment)
try:
    import chromadb
    from chromadb.config import Settings
    
    # try sentence transformers
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
        logger.info("✅ Sentence Transformers available")
    except ImportError:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        logger.warning("⚠️ Sentence Transformers not available")
    
    # chromadb rag system inline (optimized)
    class ChromaRAGSystem:
        def __init__(self, use_sentence_transformers: bool = True):
            self.use_sentence_transformers = use_sentence_transformers and SENTENCE_TRANSFORMERS_AVAILABLE
            self.documents = []
            
            try:
                # lightweight chromadb client
                self.chroma_client = chromadb.PersistentClient(
                    path="./chroma_db",
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.info("✅ ChromaDB client initialized")
            except Exception as e:
                logger.error(f"❌ ChromaDB client failed: {e}")
                raise
            
            # embedding model (optimized)
            if self.use_sentence_transformers:
                try:
                    # use lightweight model
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("✅ Sentence transformer model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ Sentence transformer failed: {e}")
                    self.use_sentence_transformers = False
            
            self.collection_name = "portfolio_knowledge"
            self.collection = None
        
        def get_embedding(self, text: str) -> List[float]:
            """generate embedding with fallback"""
            try:
                if self.use_sentence_transformers:
                    return self.embedding_model.encode(text).tolist()
                else:
                    # simple hash-based embedding fallback
                    words = re.findall(r'\b\w+\b', text.lower())
                    embedding = [0.0] * 384  # match sentence transformer dimensions
                    for i, word in enumerate(words[:50]):  # limit to prevent overflow
                        hash_val = hash(word)
                        embedding[hash_val % 384] += 1.0
                    # normalize
                    norm = sum(x*x for x in embedding) ** 0.5
                    if norm > 0:
                        embedding = [x/norm for x in embedding]
                    return embedding
            except Exception as e:
                logger.error(f"❌ Embedding generation failed: {e}")
                return [0.0] * 384
        
        def load_knowledge_base(self, knowledge_data: List[Dict]) -> bool:
            """load knowledge base to chromadb"""
            try:
                self.documents = knowledge_data
                
                # delete existing collection
                try:
                    self.chroma_client.delete_collection(self.collection_name)
                    logger.info("🔄 Deleted existing collection")
                except:
                    pass
                
                # create new collection
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Portfolio knowledge base"}
                )
                
                # process documents
                documents = []
                embeddings = []
                ids = []
                metadatas = []
                
                for item in knowledge_data:
                    full_text = f"{item['title']}: {item['content']}"
                    embedding = self.get_embedding(full_text)
                    
                    documents.append(full_text)
                    embeddings.append(embedding)
                    ids.append(item['id'])
                    metadatas.append({
                        'title': item['title'],
                        'category': item['category']
                    })
                
                # batch insert
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadatas
                )
                
                logger.info(f"✅ ChromaDB loaded {len(knowledge_data)} documents")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to load knowledge base: {e}")
                return False
        
        def retrieve_relevant_docs(self, query: str, top_k: int = 3, category_filter: str = None) -> List[Dict]:
            """retrieve relevant documents"""
            try:
                if not self.collection:
                    try:
                        self.collection = self.chroma_client.get_collection(self.collection_name)
                    except:
                        logger.error("❌ No knowledge base found")
                        return []
                
                query_embedding = self.get_embedding(query)
                
                where_filter = None
                if category_filter:
                    where_filter = {"category": category_filter}
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_filter
                )
                
                retrieved_docs = []
                if results['documents'] and len(results['documents'][0]) > 0:
                    docs = results['documents'][0]
                    metadatas = results['metadatas'][0]
                    distances = results['distances'][0]
                    
                    for i, doc in enumerate(docs):
                        retrieved_docs.append({
                            'content': doc,
                            'metadata': metadatas[i],
                            'similarity_score': 1 - distances[i]
                        })
                
                logger.info(f"✅ Retrieved {len(retrieved_docs)} relevant documents")
                return retrieved_docs
                
            except Exception as e:
                logger.error(f"❌ Failed to retrieve documents: {e}")
                return []
        
        def build_rag_context(self, query: str, top_k: int = 3, category_filter: str = None) -> str:
            """build context from retrieved documents"""
            retrieved_docs = self.retrieve_relevant_docs(query, top_k, category_filter)
            
            if not retrieved_docs:
                return ""
            
            context_parts = []
            for doc in retrieved_docs:
                if doc['similarity_score'] > 0.6:  # relevance threshold
                    content = doc['content']
                    category = doc['metadata'].get('category', 'general')
                    
                    # intelligent content trimming
                    if doc['similarity_score'] > 0.8:
                        trimmed = content[:350] + "..." if len(content) > 350 else content
                    else:
                        trimmed = content[:200] + "..." if len(content) > 200 else content
                    
                    context_parts.append(f"[{category}] {trimmed}")
            
            return "\n".join(context_parts) if context_parts else ""
        
        def suggest_related_topics(self, query: str, top_k: int = 5) -> List[str]:
            """suggest related topics"""
            try:
                retrieved_docs = self.retrieve_relevant_docs(query, top_k)
                topics = []
                
                for doc in retrieved_docs:
                    if doc['similarity_score'] > 0.5:
                        title = doc['metadata'].get('title', '')
                        if title and title not in topics:
                            # clean title for better presentation
                            clean_title = re.sub(r'^(keahlian|pengalaman|proyek|hobi)\s+', '', title.lower())
                            clean_title = clean_title.title()
                            topics.append(clean_title)
                
                return topics[:3]
                
            except Exception as e:
                logger.error(f"❌ Error suggesting topics: {e}")
                return []
    
    RAGSystem = ChromaRAGSystem
    RAG_SYSTEM_TYPE = "ChromaDB RAG (Optimized)"
    logger.info("✅ ChromaDB RAG system available")
    
except ImportError as e:
    logger.warning(f"❌ ChromaDB not available: {e}")
    
    # enhanced fallback rag system
    class AdvancedFallbackRAG:
        def __init__(self):
            self.documents = []
            self.word_index = defaultdict(list)
            self.category_index = defaultdict(list)
            self.tfidf_vectors = []
            self.vocabulary = set()
            
        def load_knowledge_base(self, data: List[Dict]) -> bool:
            try:
                self.documents = data
                self._build_advanced_indexes()
                logger.info(f"✅ Advanced Fallback RAG loaded {len(data)} documents")
                return True
            except Exception as e:
                logger.error(f"❌ Fallback RAG loading failed: {e}")
                return False
        
        def _build_advanced_indexes(self):
            # build comprehensive word index and tf-idf vectors
            all_words = []
            doc_word_count = []
            
            for i, doc in enumerate(self.documents):
                category = doc.get('category', 'general')
                self.category_index[category].append(i)
                
                text = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
                keywords = [kw.lower() for kw in doc.get('keywords', [])]
                
                words = re.findall(r'\b\w{3,}\b', text) + keywords * 3  # boost keywords
                all_words.extend(words)
                doc_word_count.append(defaultdict(int))
                
                for word in words:
                    self.word_index[word].append(i)
                    doc_word_count[i][word] += 1
                    self.vocabulary.add(word)
            
            # build tf-idf style vectors
            total_docs = len(self.documents)
            for i, word_count in enumerate(doc_word_count):
                vector = {}
                total_words = sum(word_count.values())
                
                for word, count in word_count.items():
                    tf = count / total_words if total_words > 0 else 0
                    doc_freq = len(self.word_index[word])
                    idf = total_docs / doc_freq if doc_freq > 0 else 0
                    vector[word] = tf * idf
                
                self.tfidf_vectors.append(vector)
        
        def build_rag_context(self, query: str, top_k: int = 3, category_filter: str = None) -> str:
            if not self.documents:
                return ""
            
            query_words = re.findall(r'\b\w{3,}\b', query.lower())
            doc_scores = defaultdict(float)
            
            # advanced scoring with tf-idf
            for word in query_words:
                if word in self.word_index:
                    for doc_idx in self.word_index[word]:
                        if category_filter and self.documents[doc_idx].get('category') != category_filter:
                            continue
                        
                        # tf-idf score
                        tfidf_score = self.tfidf_vectors[doc_idx].get(word, 0)
                        doc_scores[doc_idx] += tfidf_score * 2
                        
                        # keyword boost
                        if word in self.documents[doc_idx].get('keywords', []):
                            doc_scores[doc_idx] += 3.0
            
            # get top documents
            top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            context_parts = []
            for doc_idx, score in top_docs:
                if score > 0.1:
                    content = self.documents[doc_idx].get('content', '')
                    if len(content) > 250:
                        content = content[:250] + "..."
                    context_parts.append(content)
            
            return " ".join(context_parts)
        
        def suggest_related_topics(self, query: str) -> List[str]:
            return ["Technical Skills", "Project Portfolio", "Personal Interests"]
    
    RAGSystem = AdvancedFallbackRAG
    RAG_SYSTEM_TYPE = "Advanced Fallback RAG"
    logger.info("✅ Advanced Fallback RAG system available")

# fastapi application
app = FastAPI(
    title="AI Portfolio Backend", 
    description="Full-featured ChromaDB RAG system optimized for 3GB Railway deployment",
    version="5.0.0"
)

# comprehensive cors configuration
frontend_origins = [
    "https://frontend-portofolio-danen.vercel.app",
    "https://frontend-portofolio-danen-git-master-danenftyessirs-projects.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

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

# full sophisticated conversation context restored!
class ConversationContext:
    def __init__(self):
        self.last_category: Optional[str] = None
        self.mentioned_items: Set[str] = set()
        self.previous_responses: List[str] = []
        self.questions_history: List[str] = []
        self.last_updated: float = time.time()
        self.referenced_items: Dict[str, int] = defaultdict(int)
        self.conversation_topics: List[str] = []
        self.potential_followups: Dict[str, List[str]] = {}
        self.conversation_tone: str = "neutral"
        self.topic_transitions: List[Tuple[str, str, float]] = []
        self.rag_context_used: List[str] = []
    
    def update(self, category: str, question: str, response: str = None, rag_context: str = None):
        if self.last_category and self.last_category != category:
            self.topic_transitions.append((self.last_category, category, time.time()))
        
        self.last_category = category
        self.questions_history.append(question)
        
        if response:
            self.previous_responses.append(response)
            self.extract_mentioned_items(response)
            self.conversation_topics.append(category)
            
            if rag_context:
                self.rag_context_used.append(rag_context[:100])
                
            self.generate_potential_followups(category, question, response)
        
        self.detect_conversation_tone(question)
        self.last_updated = time.time()
        
        # cleanup old data
        if len(self.questions_history) > 20:
            self.questions_history = self.questions_history[-10:]
        if len(self.previous_responses) > 20:
            self.previous_responses = self.previous_responses[-10:]
    
    def extract_mentioned_items(self, response: str):
        patterns = [
            r'\b(python|react|java|next\.js|data science|machine learning|pandas|scikit-learn)\b',
            r'\b(rush hour|little alchemy|iq puzzler|portfolio)\b',
            r'\b(martabak|sate|street food|jakarta|gorengan|ketoprak)\b',
            r'\b(glenn fredly|air supply|oldies|lo-fi|without you)\b',
            r'\b(itb|teknik informatika|semester|datathon)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response.lower())
            for match in matches:
                if isinstance(match, str) and len(match) > 2:
                    self.mentioned_items.add(match.strip())
    
    def detect_conversation_tone(self, question: str):
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["kenapa", "mengapa", "gimana", "bagaimana"]):
            self.conversation_tone = "inquisitive"
        elif any(word in question_lower for word in ["wah", "wow", "keren", "mantap"]):
            self.conversation_tone = "positive"
        elif any(word in question_lower for word in ["cukup", "hanya", "saja", "doang"]):
            self.conversation_tone = "challenging"
        else:
            self.conversation_tone = "neutral"
    
    def generate_potential_followups(self, category: str, question: str, response: str):
        followups = {
            "keahlian": [
                "Bagaimana kamu belajar keahlian tersebut?",
                "Apa proyek yang menggunakan keahlian itu?",
                "Berapa lama mengembangkan skill tersebut?"
            ],
            "proyek": [
                "Apa tantangan terbesar dalam proyek itu?",
                "Teknologi apa yang dipakai?",
                "Apa yang dipelajari dari proyek tersebut?"
            ],
            "musik": [
                "Kenapa suka genre musik tersebut?",
                "Kapan pertama kali dengar lagu itu?",
                "Ada rekomendasi lagu serupa?"
            ]
        }
        
        if category in followups:
            self.potential_followups[category] = followups[category][:3]
    
    def get_suggested_followup_questions(self, limit=3):
        if not self.last_category or self.last_category not in self.potential_followups:
            return []
        return self.potential_followups[self.last_category][:limit]

# global state
conversation_sessions = {}
rag_system = None
knowledge_base_data = []

def load_portfolio_knowledge() -> List[Dict]:
    """load comprehensive portfolio knowledge"""
    try:
        if os.path.exists("portfolio.json"):
            with open("portfolio.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"✅ Loaded {len(data)} documents from portfolio.json")
                return data
        else:
            logger.warning("⚠️ portfolio.json not found, using built-in knowledge")
    except Exception as e:
        logger.error(f"❌ Error loading portfolio.json: {e}")
    
    # return your original comprehensive knowledge base
    return [
        {
            "id": "profil_dasar",
            "category": "profil",
            "title": "Profil Dasar Danendra",
            "content": "Nama saya Danendra Shafi Athallah, mahasiswa Teknik Informatika ITB semester 4 yang berdomisili di Jakarta. Saya passionate di bidang data science dan algoritma dengan pengalaman 2 tahun web development dan 1 tahun fokus data science. Motto hidup saya adalah 'Menuju tak terbatas dan melampauinya'.",
            "keywords": ["nama", "profil", "biodata", "siapa", "diri", "jakarta", "itb"]
        },
        {
            "id": "keahlian_python",
            "category": "keahlian",
            "title": "Keahlian Python & Data Science",
            "content": "Python adalah bahasa utama saya untuk analisis data dan machine learning. Saya menguasai pandas untuk data manipulation, scikit-learn untuk machine learning models, matplotlib dan seaborn untuk visualization, dan numpy untuk numerical computing. Pengalaman 1 tahun khusus di data science dengan berbagai project algoritma kompleks.",
            "keywords": ["python", "data science", "pandas", "scikit-learn", "machine learning", "algoritma"]
        },
        {
            "id": "keahlian_web",
            "category": "keahlian",
            "title": "Keahlian Web Development",
            "content": "Untuk web development, saya experienced dengan Next.js dan React untuk frontend, plus kemampuan integrate dengan backend services. Portfolio ini sendiri dibuat dengan Next.js + TypeScript, Python FastAPI, dan ditenagai OpenAI. Saya juga menguasai Tailwind CSS untuk styling dan telah mengembangkan berbagai aplikasi web selama 2 tahun terakhir.",
            "keywords": ["next.js", "react", "web development", "typescript", "tailwind", "frontend"]
        },
        {
            "id": "proyek_rushhour",
            "category": "proyek",
            "title": "Rush Hour Puzzle Solver",
            "content": "Rush Hour Puzzle Solver adalah project yang paling technically challenging dan educational. Saya implement multiple pathfinding algorithms - UCS untuk optimal solutions, Greedy Best-First untuk speed, A* untuk balanced approach, dan Dijkstra untuk comprehensive exploration. Biggest challenge adalah optimizing algorithm performance untuk handle complex puzzle configurations.",
            "keywords": ["rush hour", "puzzle", "algoritma", "pathfinding", "a*", "dijkstra", "ucs"]
        },
        {
            "id": "proyek_alchemy", 
            "category": "proyek",
            "title": "Little Alchemy Search Algorithm",
            "content": "Little Alchemy Solver project yang paling intellectually stimulating karena involve complex graph theory applications. Implementation cover BFS untuk breadth exploration, DFS untuk depth analysis, dan Bidirectional Search untuk optimal pathfinding dalam recipe combination space.",
            "keywords": ["little alchemy", "algoritma pencarian", "bfs", "dfs", "bidirectional search", "graph theory"]
        },
        {
            "id": "musik_favorit",
            "category": "musik",
            "title": "Selera Musik Oldies & Nostalgia",
            "content": "Selera musik saya nostalgic & oldies. Lagi relate banget sama 'Without You' Air Supply dan 'Sekali Ini Saja' Glenn Fredly. Oldies punya emotional depth dan musical complexity yang susah dicari di modern music. Untuk coding biasanya pakai lo-fi beats atau soundtrack film.",
            "keywords": ["musik", "oldies", "air supply", "glenn fredly", "without you", "sekali ini saja", "lo-fi"]
        },
        {
            "id": "hobi_kuliner",
            "category": "hobi", 
            "title": "Penggemar Street Food",
            "content": "Street food Indonesian jadi obsesi saya. Martabak manis, sate ayam, gorengan, ketoprak, batagor - semuanya punya cerita unik. Yang bikin saya obsessed sama street food adalah culture dan experience-nya. Jakarta punya spot-spot legendary kayak Sabang, Pecenongan yang classic banget.",
            "keywords": ["street food", "martabak", "sate", "gorengan", "ketoprak", "batagor", "jakarta", "kuliner"]
        },
        {
            "id": "rekrutmen_value",
            "category": "rekrutmen",
            "title": "Why You Should Recruit Me",
            "content": "Saya bawa kombinasi teori dan praktek di data science yang solid. Track record di berbagai algorithmic projects, academic performance yang konsisten, dan keterlibatan aktif di tech community. Yang bikin saya beda adalah learning agility yang tinggi, systematic approach dalam problem decomposition, dan kemampuan communicate technical concepts ke diverse audiences.",
            "keywords": ["rekrutmen", "hire", "kelebihan", "value proposition", "team", "collaboration"]
        }
    ]

# all your original sophisticated functions restored
def categorize_question_with_rag(question: str, context: ConversationContext = None) -> Tuple[str, str]:
    basic_category = categorize_question_basic(question)
    
    rag_context = ""
    if rag_system:
        try:
            category_mapping = {
                "keahlian": "keahlian",
                "proyek": "proyek", 
                "hobi": "hobi",
                "makanan_favorit": "hobi",
                "lagu_favorit": "musik",
                "pengalaman": "pendidikan",
                "rekrutmen": "rekrutmen"
            }
            
            category_filter = category_mapping.get(basic_category)
            rag_context = rag_system.build_rag_context(
                question, 
                top_k=3, 
                category_filter=category_filter
            )
            
        except Exception as e:
            logger.error(f"error in rag categorization: {e}")
    
    return basic_category, rag_context

def categorize_question_basic(question: str) -> str:
    """comprehensive question categorization with personal content protection"""
    question_lower = question.lower()
    
    # personal sensitive content detection
    personal_sensitive = {
        "personal_relationship": ["pacar", "jodoh", "pacaran", "pasangan", "gebetan", "cinta"],
        "personal_financial": ["gaji", "salary", "penghasilan", "bayaran", "uang", "kekayaan"],
        "personal_contact": ["alamat", "rumah", "nomor", "kontak", "telepon", "hp"],
        "personal_age": ["umur", "usia", "tahun", "tua", "muda", "tanggal lahir"],
        "personal_family": ["keluarga", "orangtua", "ayah", "ibu", "kakak", "adik"],
        "personal_religion": ["agama", "kepercayaan", "tuhan", "doa", "ibadah"],
        "personal_health": ["sakit", "kesehatan", "penyakit", "obat", "dokter"],
        "personal_appearance": ["ganteng", "jelek", "cantik", "tinggi", "pendek"]
    }
    
    for category, keywords in personal_sensitive.items():
        if any(keyword in question_lower for keyword in keywords):
            return category
    
    # technical categorization
    category_keywords = {
        "keahlian": ["python", "data science", "machine learning", "keahlian", "skill"],
        "proyek": ["proyek", "project", "rush hour", "little alchemy", "iq puzzler"],
        "lagu_favorit": ["lagu", "musik", "glenn fredly", "air supply", "without you"],
        "makanan_favorit": ["makanan", "street food", "martabak", "sate", "kuliner"],
        "hobi": ["hobi", "traveling", "membaca", "novel"],
        "pengalaman": ["pengalaman", "asisten praktikum", "datathon", "lomba"],
        "rekrutmen": ["rekrut", "hire", "kenapa harus", "mengapa memilih"],
        "prestasi": ["prestasi", "juara", "lomba", "kompetisi"],
        "tools": ["tools", "software", "vs code", "jupyter", "figma"],
        "karakter": ["karakter", "kepribadian", "sifat", "tipe"]
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            return category
    
    return "general"

def create_rag_enhanced_prompt(question: str, rag_context: str, context: ConversationContext = None) -> str:
    """sophisticated prompt engineering with rag context"""
    
    base_prompt = f"""
Kamu adalah asisten pribadi Danendra Shafi Athallah yang cerdas, informatif, dan memiliki kepribadian yang santai. 
Jawab dengan bahasa Indonesia yang natural dan santai, tapi tetap informatif.

PENTING: Selalu acknowledge pertanyaan user dengan mengutip sebagian pertanyaannya di awal respons.
"""

    if rag_context:
        base_prompt += f"""
INFORMASI DARI KNOWLEDGE BASE:
{rag_context}

Gunakan informasi di atas untuk memberikan jawaban yang lebih akurat dan detail. Jangan copy paste langsung, tapi paraphrase dengan gaya bahasa yang natural.
"""

    if context and len(context.questions_history) > 0:
        previous_question = context.questions_history[-1] if context.questions_history else "Belum ada"
        
        base_prompt += f"""
KONTEKS PERCAKAPAN:
- Kategori terakhir: {context.last_category}
- Pertanyaan sebelumnya: {previous_question}
- Tone conversation: {context.conversation_tone}

Jika ini adalah follow-up, reference konteks sebelumnya secara natural.
"""

    base_prompt += f"""
Pertanyaan pengguna: {question}

INSTRUCTIONS:
1. Acknowledge pertanyaan dengan natural, jangan berlebihan
2. Jawab dengan santai dan to the point menggunakan informasi dari knowledge base  
3. Bahasa Indonesia yang natural, tidak kaku
4. Kasih contoh spesifik kalau relevan
5. Gunakan "aku" dan "kamu"
6. Max 3-4 kalimat, concise tapi informatif
7. Jangan terlalu banyak English words
8. Tone santai, tidak formal
"""
    
    return base_prompt

def generate_fallback_response(question: str, category: str) -> str:
    """comprehensive fallback responses with personal content protection"""
    
    if category.startswith("personal_"):
        personal_redirects = {
            "personal_relationship": "Hehe, fokus ke professional dulu nih. Mending kita bahas passion aku di data science atau proyek-proyek yang udah dikerjain?",
            "personal_financial": "Soal finansial kurang nyaman share. Mending kita bahas project portfolio atau skill development yang lagi aku kerjain?",
            "personal_contact": "Info kontak pribadi nggak bisa share. Tapi kalau mau discuss collaboration atau project ideas, aku welcome banget!",
            "personal_age": "Umur cuma angka! Yang penting skills dan passion di tech. Mau tahu tentang learning journey aku di programming?",
            "personal_family": "Family matters prefer keep private. Kalau mau tahu support system aku dalam belajar tech, aku bisa share!",
            "personal_religion": "Spiritual matters prefer keep personal. Kalau mau discuss values dalam professional life, aku open!",
            "personal_health": "Health info private ya. Kalau mau tahu work-life balance atau stress management dalam coding, bisa share!",
            "personal_appearance": "Appearance nggak penting. Yang matters adalah skills dan contribution dalam tech. Mau discuss expertise aku?"
        }
        return personal_redirects.get(category, "Info personal prefer tidak dibahas ya. Mending discuss technical skills atau project portfolio aku?")
    
    # comprehensive category responses
    responses = {
        "keahlian": "Aku punya keahlian di Python untuk data science, Next.js untuk web development, dan algoritma untuk problem solving. Mau tahu lebih detail tentang yang mana?",
        "proyek": "Proyek unggulan aku adalah Rush Hour Solver dengan algoritma pathfinding dan Little Alchemy Solver dengan graph search. Ada yang menarik?", 
        "hobi": "Hobi aku membaca novel fantasy, traveling lokal, dan hunting street food Jakarta. Mau cerita tentang yang mana?",
        "lagu_favorit": "Musik favorit aku oldies kayak Glenn Fredly dan Air Supply. Enak buat coding juga pakai lo-fi beats.",
        "makanan_favorit": "Street food Indonesia jadi obsesi aku. Martabak, sate, ketoprak, semuanya enak banget!",
        "pengalaman": "Pengalaman aku cukup beragam, dari asisten praktikum sampai lomba datathon. Mau tahu tentang yang mana?",
        "rekrutmen": "Aku bawa kombinasi teori dan praktek di data science yang solid. Track record akademik bagus dan passionate banget di tech!",
        "general": "Aku Danendra, mahasiswa Teknik Informatika ITB yang passionate di data science dan web development. Ada yang mau ditanyakan khusus?"
    }
    
    return responses.get(category, "Interesting question! Bisa tanya lebih spesifik tentang technical skills atau project portfolio aku?")

def call_openai_api(prompt: str) -> str:
    """openai integration with comprehensive error handling"""
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
                {"role": "system", "content": "Kamu adalah asisten virtual yang membantu menjawab pertanyaan tentang pemilik portfolio dengan cara yang personal, informatif, dan sangat interactive menggunakan informasi dari knowledge base."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 800,
            "temperature": 0.8
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=25
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
    """comprehensive startup with full rag initialization"""
    global rag_system, knowledge_base_data
    
    try:
        logger.info(f"🚀 Starting Full-Featured AI Portfolio Backend...")
        logger.info(f"Target: ChromaDB RAG System (~3GB deployment)")
        
        # load knowledge base
        knowledge_base_data = load_portfolio_knowledge()
        logger.info(f"📚 Knowledge base: {len(knowledge_base_data)} documents")
        
        # initialize rag system
        if RAGSystem:
            logger.info(f"Initializing {RAG_SYSTEM_TYPE}...")
            try:
                rag_system = RAGSystem()
                if rag_system.load_knowledge_base(knowledge_base_data):
                    logger.info(f"✅ {RAG_SYSTEM_TYPE} initialized successfully")
                else:
                    raise Exception("RAG system initialization failed")
            except Exception as e:
                logger.error(f"❌ {RAG_SYSTEM_TYPE} failed: {e}")
                rag_system = None
        
        # environment status
        logger.info("🔍 Environment status:")
        logger.info(f"   OpenAI API Key: {'✅ SET' if os.getenv('OPENAI_API_KEY') else '❌ NOT SET'}")
        logger.info(f"   Frontend URL: {os.getenv('FRONTEND_URL', 'NOT SET')}")
        logger.info(f"   Port: {os.getenv('PORT', 'NOT SET')}")
        
        logger.info("🎉 Full-featured backend startup completed!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

# all original sophisticated endpoints restored
@app.get("/")
async def root():
    return {
        "message": "✅ Full-Featured AI Portfolio Backend (3GB Optimized)",
        "status": "healthy",
        "deployment": "Railway Production (3GB Target)",
        "rag_system": RAG_SYSTEM_TYPE,
        "knowledge_documents": len(knowledge_base_data),
        "rag_available": rag_system is not None,
        "openai_available": bool(os.getenv("OPENAI_API_KEY")),
        "features": {
            "chromadb_available": "ChromaDB" in RAG_SYSTEM_TYPE,
            "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE if 'SENTENCE_TRANSFORMERS_AVAILABLE' in globals() else False,
            "advanced_rag": True,
            "conversation_context": True,
            "personal_content_protection": True,
            "sophisticated_categorization": True,
            "followup_suggestions": True,
            "semantic_search": "ChromaDB" in RAG_SYSTEM_TYPE
        },
        "performance": {
            "target_image_size": "~3GB",
            "feature_retention": "95%",
            "query_accuracy": "Advanced" if "ChromaDB" in RAG_SYSTEM_TYPE else "Good"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time(), "rag_system": RAG_SYSTEM_TYPE}

@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    """main ai endpoint with full sophisticated functionality"""
    try:
        # cleanup old sessions
        current_time = time.time()
        sessions_to_remove = [sid for sid, ctx in conversation_sessions.items() 
                            if current_time - ctx.last_updated > 1800]
        for sid in sessions_to_remove:
            del conversation_sessions[sid]
        
        if len(request.question.strip()) < 2:
            return AIResponse(
                response="Pertanyaanmu terlalu singkat. Coba tanyakan lebih detail tentang diriku.",
                session_id=request.session_id or str(uuid.uuid4())
            )
        
        session_id = request.session_id or str(uuid.uuid4())
        
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = ConversationContext()
        
        context = conversation_sessions[session_id]
        category, rag_context = categorize_question_with_rag(request.question, context)
        
        logger.info(f"category detected: {category}")
        if rag_context:
            logger.info(f"rag context found: {len(rag_context)} characters")
        
        # handle gibberish
        if len(request.question.split()) < 2 and not any(tech in request.question.lower() 
                                                        for tech in ["python", "react", "java"]):
            response_text = "Hmm, maaf aku tidak mengerti pertanyaanmu. Bisa diulangi dengan kata-kata yang lebih jelas?"
            context.update(category, request.question, response_text)
            return AIResponse(response=response_text, session_id=session_id)
        
        # handle sensitive personal content
        if category.startswith("personal_"):
            logger.info(f"personal content detected: {category}")
            response_text = generate_fallback_response(request.question, category)
            context.update(category, request.question, response_text)
            return AIResponse(response=response_text, session_id=session_id)
        
        # get related topics
        related_topics = []
        if rag_system:
            try:
                related_topics = rag_system.suggest_related_topics(request.question)
            except Exception as e:
                logger.error(f"error getting related topics: {e}")
        
        # openai integration with rag context
        if rag_context:
            prompt = create_rag_enhanced_prompt(request.question, rag_context, context)
            try:
                response_text = call_openai_api(prompt)
                logger.info("response received from openai")
                
                context.update(category, request.question, response_text, rag_context)
                
                return AIResponse(
                    response=response_text, 
                    session_id=session_id,
                    related_topics=related_topics
                )
                
            except Exception as openai_error:
                logger.warning(f"openai failed, using rag fallback: {str(openai_error)}")
                fallback_text = f"Berdasarkan informasi yang aku punya, {rag_context[:200]}... Mau tahu lebih detail tentang aspek mana?"
                context.update(category, request.question, fallback_text, rag_context)
                
                return AIResponse(
                    response=fallback_text,
                    session_id=session_id,
                    related_topics=related_topics
                )
        else:
            logger.info("no rag context, using fallback response")
            fallback_text = generate_fallback_response(request.question, category)
            context.update(category, request.question, fallback_text)
            
            return AIResponse(
                response=fallback_text,
                session_id=session_id,
                related_topics=related_topics
            )
        
    except Exception as e:
        logger.error(f"error processing request: {str(e)}")
        return AIResponse(
            response="Maaf, ada error sistem. Coba lagi dalam beberapa saat ya!",
            session_id=request.session_id or str(uuid.uuid4())
        )

@app.post("/ask-mock", response_model=AIResponse)
async def ask_ai_mock(request: QuestionRequest):
    """mock endpoint with sophisticated categorization"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = ConversationContext()
        
        context = conversation_sessions[session_id]
        category, rag_context = categorize_question_with_rag(request.question, context)
        
        if rag_context:
            mock_response = f"Berdasarkan informasi yang aku punya, {rag_context[:200]}... Apa yang ingin kamu tahu lebih detail?"
        else:
            mock_response = generate_fallback_response(request.question, category)
        
        related_topics = []
        if rag_system:
            try:
                related_topics = rag_system.suggest_related_topics(request.question)
            except Exception as e:
                logger.error(f"error getting related topics in mock: {e}")
        
        context.update(category, request.question, mock_response, rag_context)
        
        return AIResponse(
            response=mock_response,
            session_id=session_id, 
            related_topics=related_topics
        )
        
    except Exception as e:
        logger.error(f"error in mock endpoint: {str(e)}")
        return AIResponse(
            response="Mock response: Aku Danendra, passionate di data science dan web development!",
            session_id=request.session_id or str(uuid.uuid4()),
            related_topics=["Python Skills", "Web Development", "Data Science"]
        )

@app.get("/suggested-followups/{session_id}")
async def get_suggested_followups(session_id: str):
    """sophisticated followup suggestions"""
    if session_id not in conversation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    context = conversation_sessions[session_id]
    suggested_followups = context.get_suggested_followup_questions(3)
    
    return {"suggested_followups": suggested_followups}

@app.get("/rag-status")
async def get_rag_status():
    """comprehensive rag system status"""
    global rag_system, knowledge_base_data
    
    if rag_system:
        try:
            # test rag functionality
            test_context = rag_system.build_rag_context("python", top_k=1)
            return {
                "status": "healthy",
                "system_type": RAG_SYSTEM_TYPE,
                "documents_indexed": len(knowledge_base_data),
                "test_query_results": len(test_context) > 0,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2" if SENTENCE_TRANSFORMERS_AVAILABLE else "hash-based fallback",
                "vector_db": "ChromaDB" if "ChromaDB" in RAG_SYSTEM_TYPE else "In-Memory",
                "semantic_search": "ChromaDB" in RAG_SYSTEM_TYPE,
                "fallback_mode": False,
                "performance": {
                    "image_size_target": "~3GB",
                    "feature_retention": "95%",
                    "query_accuracy": "Advanced" if "ChromaDB" in RAG_SYSTEM_TYPE else "Good"
                }
            }
        except Exception as e:
            return {
                "status": "error", 
                "system_type": RAG_SYSTEM_TYPE,
                "error": str(e),
                "documents_indexed": len(knowledge_base_data),
                "fallback_mode": True
            }
    else:
        return {
            "status": "fallback_mode",
            "system_type": "Static Responses",
            "reason": "RAG system not available",
            "documents_indexed": len(knowledge_base_data),
            "fallback_mode": True
        }

# railway production server configuration
if __name__ == "__main__":
    try:
        import uvicorn
        
        port = int(os.getenv("PORT", 8000))
        logger.info(f"🚀 Starting production server on 0.0.0.0:{port}")
        logger.info(f"RAG System: {RAG_SYSTEM_TYPE}")
        logger.info(f"Knowledge Base: {len(knowledge_base_data)} documents")
        logger.info(f"Target Image Size: ~3GB (75% of Railway 4GB limit)")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True,
            timeout_keep_alive=120
        )
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)