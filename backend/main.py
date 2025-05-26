from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import logging
import json
import requests
import random
import re
import uuid
from typing import List, Dict, Any, Optional, Set, Tuple
import difflib
import time
from collections import defaultdict

# setup logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# smart import dengan fallback mechanism - keep original logic
RAG_SYSTEM_TYPE = "None"
RAGSystem = None
initialize_rag_system = None
load_knowledge_from_file = None

try:
    # coba import chromadb version dulu
    from rag_system import RAGSystem, initialize_rag_system, load_knowledge_from_file
    RAG_SYSTEM_TYPE = "ChromaDB"
    logger.info("✅ ChromaDB RAG system loaded successfully")
except ImportError as e:
    logger.warning(f"❌ ChromaDB not available: {e}")
    try:
        # fallback ke simple rag system
        from rag_system import SimpleRAGSystem as RAGSystem, initialize_rag_system, load_knowledge_from_file
        RAG_SYSTEM_TYPE = "Simple"
        logger.info("✅ Simple RAG system loaded as fallback")
    except ImportError as e2:
        logger.error(f"❌ No RAG system available: {e2}")
        RAG_SYSTEM_TYPE = "None"

app = FastAPI(title=f"AI Portfolio Backend - {RAG_SYSTEM_TYPE} RAG")

# keep original CORS logic
frontend_url = os.getenv("FRONTEND_URL", "https://frontend-portofolio-danen.vercel.app")

allowed_origins = []
if frontend_url:
    for url in frontend_url.split(','):
        url = url.strip()
        if url:
            allowed_origins.append(url)

additional_origins = [
    "http://localhost:3000",
    "https://frontend-portofolio-danen.vercel.app",
    "https://frontend-portofolio-danen-git-master-danenftyessirs-projects.vercel.app",
    "https://frontend-portofolio-danen-419wreixy-danenftyessirs-projects.vercel.app"
]

for origin in additional_origins:
    if origin not in allowed_origins:
        allowed_origins.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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

# keep full original ConversationContext - all sophisticated features intact
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
        self.last_response_category: Optional[str] = None
        self.rag_context_used: List[str] = []
    
    def update(self, category: str, question: str, response: str = None, rag_context: str = None):
        if self.last_category and self.last_category != category:
            self.topic_transitions.append((self.last_category, category, time.time()))
            if category not in ["general", "unclear_question", "gibberish"]:
                self.potential_followups.clear()
        
        self.last_category = category
        self.questions_history.append(question)
        
        if response:
            self.previous_responses.append(response)
            self.last_response_category = category
            self.extract_mentioned_items(response)
            
            if category not in ["unclear_question", "general"]:
                self.conversation_topics.append(category)
            
            if rag_context:
                self.rag_context_used.append(rag_context[:100])
                
            if not self.topic_transitions or self.topic_transitions[-1][1] == category:
                self.generate_potential_followups(category, question, response)
        
        self.detect_conversation_tone(question)
        
        for item in self.mentioned_items:
            if re.search(r'\b' + re.escape(item) + r'\b', question.lower()):
                self.referenced_items[item] += 1
        
        self.last_updated = time.time()
    
    def is_context_relevant(self, new_category: str) -> bool:
        if not self.last_category:
            return False
        
        if new_category == self.last_category:
            return True
        
        related_categories = {
            'keahlian': ['teknologi', 'proyek', 'tools', 'pengalaman'],
            'teknologi': ['keahlian', 'proyek'],
            'proyek': ['keahlian', 'teknologi', 'pengalaman'], 
            'pengalaman': ['keahlian', 'proyek', 'pendidikan'],
            'hobi': ['personal', 'musik', 'lifestyle'],
            'personal': ['hobi', 'karakter', 'lifestyle'],
            'musik': ['hobi', 'personal'],
            'prestasi': ['pengalaman', 'pendidikan'],
            'rekrutmen': ['keahlian', 'proyek', 'karakter']
        }
        
        if self.last_category in related_categories:
            if new_category in related_categories[self.last_category]:
                return True
        
        if new_category.endswith('_followup'):
            base_category = new_category.replace('_followup', '')
            return base_category == self.last_category
        
        return False
    
    def should_use_context_in_prompt(self, new_category: str) -> bool:
        if not self.is_context_relevant(new_category):
            return False
        
        if len(self.topic_transitions) > 3:
            return False
        
        if self.topic_transitions and (time.time() - self.topic_transitions[-1][2]) > 300:
            return False
        
        return True
    
    def detect_conversation_tone(self, question: str):
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["kenapa", "mengapa", "gimana", "bagaimana", "kok", "masa"]):
            self.conversation_tone = "inquisitive"
        elif any(word in question_lower for word in ["wah", "wow", "keren", "mantap", "bagus"]):
            self.conversation_tone = "positive"
        elif any(word in question_lower for word in ["cukup", "hanya", "saja", "doang", "emang", "yakin"]):
            self.conversation_tone = "challenging"
        elif any(word in question_lower for word in ["bisa", "tolong", "help", "bantu"]):
            self.conversation_tone = "seeking_help"
        else:
            self.conversation_tone = "neutral"
    
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
            "hobi": [
                "Sejak kapan suka hal tersebut?",
                "Apa yang menarik dari hobi itu?",
                "Seberapa sering melakukan aktivitas tersebut?"
            ],
            "musik": [
                "Kenapa suka genre musik tersebut?",
                "Kapan pertama kali dengar lagu itu?",
                "Ada rekomendasi lagu serupa?"
            ],
            "pengalaman": [
                "Cerita lebih detail tentang pengalaman itu?",
                "Apa insight yang didapat?",
                "Bagaimana pengalaman itu membentuk skill?"
            ]
        }
        
        items_in_response = []
        for item in self.mentioned_items:
            if item in response.lower():
                items_in_response.append(item)
                
        if category in followups and len(self.questions_history) < 5:
            self.potential_followups[category] = followups[category][:3]

    def get_suggested_followup_questions(self, limit=3):
        if not self.last_category or self.last_category not in self.potential_followups:
            return []
            
        return self.potential_followups[self.last_category][:limit]

# global variables - keep original structure
conversation_sessions = {}
rag_system = None
knowledge_base_data = []

def cleanup_old_sessions():
    current_time = time.time()
    sessions_to_remove = []
    
    for session_id, context in conversation_sessions.items():
        if current_time - context.last_updated > 1800:  # 30 menit
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del conversation_sessions[session_id]
    
    if sessions_to_remove:
        logger.info(f"cleaned up {len(sessions_to_remove)} inactive sessions")

def load_portfolio_knowledge() -> List[Dict]:
    """load knowledge base dari file portfolio.json - keep original logic"""
    try:
        if os.path.exists("portfolio.json"):
            with open("portfolio.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"loaded {len(data)} documents from portfolio.json")
                return data
        else:
            logger.warning("portfolio.json not found, using fallback data")
            return [
                {
                    "id": "profil_fallback",
                    "category": "profil",
                    "title": "Profil Dasar",
                    "content": "Danendra Shafi Athallah, mahasiswa Teknik Informatika ITB yang passionate di data science dan web development."
                }
            ]
    except Exception as e:
        logger.error(f"error loading portfolio.json: {e}")
        return []

# keep all original sophisticated functions
def is_gibberish(text: str) -> bool:
    if len(text) < 2:
        return False
        
    text = text.lower()
    
    if any(tech in text for tech in ["python", "react", "java", "next.js", "data science"]):
        return False
    
    if any(key in text for key in ["ceritakan", "bagaimana", "gimana", "tentang"]):
        return False
    
    if len(text.split()) < 2:
        valid_terms = ["python", "react", "java", "next", "hobi", "makanan", "lagu", "proyek"]
        return not any(term in text for term in valid_terms)
    
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    
    max_consecutive_consonants = 0
    current_consonants = 0
    
    for char in text:
        if char in consonants:
            current_consonants += 1
            max_consecutive_consonants = max(max_consecutive_consonants, current_consonants)
        else:
            current_consonants = 0
    
    char_counts = {}
    for char in text:
        if char.isalpha():
            char_counts[char] = char_counts.get(char, 0) + 1
    
    vowel_count = sum(char_counts.get(v, 0) for v in vowels)
    total_chars = sum(char_counts.values())
    
    if max_consecutive_consonants >= 4:
        return True
    
    if total_chars > 0 and vowel_count / total_chars < 0.1:
        return True
    
    return False

def categorize_question_with_rag(question: str, context: ConversationContext = None) -> Tuple[str, str]:
    """kategorisasi pertanyaan dengan bantuan rag untuk context tambahan - keep original logic"""
    global rag_system
    
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
                "rekrutmen": "rekrutmen",
                "prestasi": "prestasi",
                "tools": "tools",
                "karakter": "karakter"
            }
            
            category_filter = category_mapping.get(basic_category)
            rag_context = rag_system.build_rag_context(
                question, 
                top_k=2, 
                category_filter=category_filter
            )
            
        except Exception as e:
            logger.error(f"error in rag categorization: {e}")
    
    return basic_category, rag_context

def categorize_question_basic(question: str) -> str:
    """basic question categorization dengan personal content filtering - keep original sophisticated logic"""
    question_lower = question.lower()
    
    if is_gibberish(question):
        return "gibberish"
    
    # personal content detection (highest priority) - keep original comprehensive list
    personal_sensitive = {
        "personal_relationship": ["pacar", "jodoh", "pacaran", "pasangan", "gebetan", "cinta", "nikah", "menikah", "single", "lajang", "relationship", "dating", "girlfriend", "boyfriend"],
        "personal_financial": ["gaji", "salary", "penghasilan", "bayaran", "uang", "kekayaan", "harta", "tabungan", "pendapatan", "income", "money", "rich", "poor"],
        "personal_contact": ["alamat", "rumah", "tinggal dimana", "nomor", "kontak", "telepon", "hp", "whatsapp", "instagram", "pribadi", "address", "phone"],
        "personal_age": ["umur", "usia", "tahun", "tua", "muda", "tanggal lahir", "kapan lahir", "kelahiran", "birthday", "age"],
        "personal_family": ["keluarga", "orangtua", "ayah", "ibu", "kakak", "adik", "saudara", "family", "parents", "father", "mother"],
        "personal_religion": ["agama", "kepercayaan", "tuhan", "doa", "ibadah", "sholat", "gereja", "religion", "pray"],
        "personal_health": ["sakit", "kesehatan", "penyakit", "obat", "dokter", "rumah sakit", "health", "sick"],
        "personal_appearance": ["ganteng", "jelek", "cantik", "tinggi", "pendek", "gemuk", "kurus", "wajah", "muka", "appearance", "looks"]
    }
    
    # check for sensitive personal content first
    for category, keywords in personal_sensitive.items():
        if any(keyword in question_lower for keyword in keywords):
            return category
    
    # regular categorization - keep original comprehensive categories
    category_keywords = {
        "keahlian": ["python", "data science", "machine learning", "pandas", "react", "next.js", "keahlian", "skill"],
        "proyek": ["proyek", "project", "rush hour", "little alchemy", "iq puzzler", "portfolio"],
        "lagu_favorit": ["lagu", "musik", "glenn fredly", "air supply", "without you", "sekali ini saja"],
        "makanan_favorit": ["makanan", "street food", "martabak", "sate", "gorengan", "ketoprak", "kuliner"],
        "hobi": ["hobi", "traveling", "membaca", "novel", "omniscient reader"],
        "pengalaman": ["pengalaman", "asisten praktikum", "datathon", "lomba", "arkavidia"],
        "rekrutmen": ["rekrut", "hire", "kenapa harus", "mengapa memilih", "kelebihan"],
        "prestasi": ["prestasi", "juara", "lomba", "kompetisi", "datathon"],
        "tools": ["tools", "software", "vs code", "jupyter", "figma"],
        "karakter": ["karakter", "kepribadian", "sifat", "tipe"],
        "pendidikan": ["itb", "kuliah", "semester", "culture shock", "pendidikan"],
        "visi": ["masa depan", "rencana", "goal", "target", "visi"],
        "lifestyle": ["stres", "manajemen", "work life balance", "produktivitas"]
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            return category
    
    if len(question.split()) < 3:
        return "unclear_question"
    else:
        return "general"

def create_rag_enhanced_prompt(question: str, rag_context: str, context: ConversationContext = None) -> str:
    """buat prompt yang enhanced dengan rag context - keep original sophisticated prompt engineering"""
    
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

    if context and context.should_use_context_in_prompt("general"):
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

Contoh yang BENAR: "Oh, soal itu aku punya pengalaman..." 
Contoh yang SALAH: "That's definitely something worth discussing..."
"""
    
    return base_prompt

def generate_fallback_response(question: str, category: str) -> str:
    """generate fallback response dengan personal content protection - keep original sophisticated responses"""
    
    # handle sensitive personal content dengan redirect yang smooth
    personal_redirects = {
        "personal_relationship": [
            "Hehe, fokus ke professional dulu nih. Mending kita bahas passion aku di data science atau proyek-proyek yang udah dikerjain?",
            "Untuk hal personal prefer nggak bahas ya. Kalau tertarik sama technical journey aku, bisa tanya tentang coding experience atau algorithm projects.",
            "Itu agak personal sih. Lebih seru discuss keahlian aku di Python dan web development. Mau tahu tentang yang mana?"
        ],
        "personal_financial": [
            "Soal finansial kurang nyaman share. Mending kita bahas project portfolio atau skill development yang lagi aku kerjain?",
            "Itu personal banget. Kalau mau tahu tentang career journey atau tech stack yang aku pakai, aku senang cerita!",
            "Financial info nggak bisa share ya. Tapi aku bisa cerita tentang investment terbesar aku - yaitu skill di data science dan programming!"
        ],
        "personal_contact": [
            "Info kontak pribadi nggak bisa share. Tapi kalau mau discuss collaboration atau project ideas, aku welcome banget!",
            "Untuk privacy nggak bisa kasih detail kontak. Kalau tertarik sama work aku, bisa check portfolio ini atau GitHub aku.",
            "Contact info nggak bisa share. Mending kita bahas technical expertise atau proyek yang lagi dikembangkan?"
        ],
        "personal_age": [
            "Umur cuma angka! Yang penting skills dan passion di tech. Mau tahu tentang learning journey aku di programming?",
            "Age is just a number. Lebih seru bahas skill development dan proyek-proyek yang udah accomplished. Interested?",
            "Soal umur kurang relevan. Yang important adalah continuous learning di tech. Mau dengar experience aku?"
        ],
        "personal_family": [
            "Family matters prefer keep private. Kalau mau tahu support system aku dalam belajar tech, aku bisa share!",
            "Info keluarga personal banget. Mending discuss mentor atau inspiration dalam tech journey aku?",
            "Family info nggak share ya. Tapi aku bisa cerita tentang community dan network di tech yang supportive!"
        ],
        "personal_religion": [
            "Spiritual matters prefer keep personal. Kalau mau discuss values dalam professional life atau team collaboration, aku open!",
            "Kepercayaan personal banget. Mending bahas values dalam coding ethics atau open source contribution?",
            "Religious views nggak discuss ya. Tapi aku bisa share tentang mindset growth dan continuous learning!"
        ],
        "personal_health": [
            "Health info private ya. Kalau mau tahu work-life balance atau stress management dalam coding, bisa share!",
            "Medical matters nggak discuss. Tapi aku bisa cerita tentang maintaining productivity dalam programming.",
            "Health personal banget. Mending bahas mental framework atau problem-solving approach aku?"
        ],
        "personal_appearance": [
            "Appearance nggak penting. Yang matters adalah skills dan contribution dalam tech. Mau discuss expertise aku?",
            "Physical looks irrelevant dalam programming. Lebih interested bahas code quality atau technical achievement?",
            "Looks doesn't matter dalam tech. Yang penting adalah logical thinking dan problem-solving ability!"
        ]
    }
    
    # kalau category adalah personal sensitive, return random redirect
    if category in personal_redirects:
        import random
        return random.choice(personal_redirects[category])
    
    # regular fallback responses untuk non-sensitive content - keep original comprehensive responses
    regular_responses = {
        "keahlian": "Aku punya keahlian di Python untuk data science, Next.js untuk web development, dan algoritma untuk problem solving. Mau tahu lebih detail tentang yang mana?",
        "proyek": "Proyek unggulan aku adalah Rush Hour Solver dengan algoritma pathfinding dan Little Alchemy Solver dengan graph search. Ada yang menarik?", 
        "hobi": "Hobi aku membaca novel fantasy, traveling lokal, dan hunting street food Jakarta. Mau cerita tentang yang mana?",
        "lagu_favorit": "Musik favorit aku oldies kayak Glenn Fredly dan Air Supply. Enak buat coding juga pakai lo-fi beats.",
        "makanan_favorit": "Street food Indonesia jadi obsesi aku. Martabak, sate, ketoprak, semuanya enak banget!",
        "pengalaman": "Pengalaman aku cukup beragam, dari asisten praktikum sampai lomba datathon. Mau tahu tentang yang mana?",
        "rekrutmen": "Aku bawa kombinasi teori dan praktek di data science yang solid. Track record akademik bagus dan passionate banget di tech!",
        "general": "Aku Danendra, mahasiswa Teknik Informatika ITB yang passionate di data science dan web development. Ada yang mau ditanyakan khusus?"
    }
    
    return regular_responses.get(category, "Interesting question! Bisa tanya lebih spesifik tentang technical skills atau project portfolio aku?")

def call_openai_api(prompt):
    logger.info("sending request to openai")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("api key not found")
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
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"openai error: {response.status_code} - {response.text}")
            raise ValueError(f"OpenAI API error: {response.status_code}")
        
        result = response.json()
        if "choices" not in result or len(result["choices"]) == 0:
            logger.error("no result from openai")
            raise ValueError("No result from OpenAI")
        
        raw_response = result["choices"][0]["message"]["content"]
        return raw_response.strip()
    
    except requests.exceptions.RequestException as e:
        logger.error(f"request error: {str(e)}")
        raise ValueError(f"Error communicating with OpenAI: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """initialize rag system saat startup dengan fallback - keep original logic"""
    global rag_system, knowledge_base_data
    try:
        logger.info(f"🚀 Starting AI Portfolio Backend with {RAG_SYSTEM_TYPE} RAG")
        logger.info("loading portfolio knowledge base...")
        knowledge_base_data = load_portfolio_knowledge()
        
        if knowledge_base_data and initialize_rag_system:
            logger.info("initializing rag system...")
            rag_system = initialize_rag_system(
                knowledge_data=knowledge_base_data,
                use_openai=False
            )
            if rag_system:
                logger.info(f"✅ {RAG_SYSTEM_TYPE} RAG system initialized successfully")
            else:
                logger.warning("❌ Failed to initialize RAG system, using fallback responses")
        else:
            logger.warning("❌ No knowledge base data found or RAG system unavailable")
            
    except Exception as e:
        logger.error(f"❌ Error in startup: {e}")
        rag_system = None

# keep ALL original endpoints with full functionality
@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    try:
        cleanup_old_sessions()
        
        logger.info(f"question received: {request.question}")
        
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
        
        if category == "gibberish":
            response_text = "Hmm, maaf aku tidak mengerti pertanyaanmu. Bisa diulangi dengan kata-kata yang lebih jelas?"
            context.update(category, request.question, response_text)
            return AIResponse(response=response_text, session_id=session_id)
        
        # handle sensitive personal content dengan protective responses
        if category.startswith("personal_"):
            logger.info(f"personal content detected: {category}")
            response_text = generate_fallback_response(request.question, category)
            context.update(category, request.question, response_text)
            return AIResponse(response=response_text, session_id=session_id)
        
        # get related topics dari rag system
        related_topics = []
        if rag_system:
            try:
                related_topics = rag_system.suggest_related_topics(request.question)
            except Exception as e:
                logger.error(f"error getting related topics: {e}")
        
        # coba openai api jika ada rag context
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
                # fallback ke rag context summary
                fallback_text = f"Berdasarkan informasi yang aku punya, {rag_context[:200]}... Mau tahu lebih detail tentang aspek mana?"
                context.update(category, request.question, fallback_text, rag_context)
                
                return AIResponse(
                    response=fallback_text,
                    session_id=session_id,
                    related_topics=related_topics
                )
        else:
            # tidak ada rag context, langsung fallback response
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
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

@app.post("/ask-mock", response_model=AIResponse)
async def ask_ai_mock(request: QuestionRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = ConversationContext()
        
        context = conversation_sessions[session_id]
        question = request.question
        
        if len(question.strip()) < 2:
            return AIResponse(
                response="Pertanyaanmu terlalu singkat. Coba tanyakan lebih detail.",
                session_id=session_id
            )
        
        if is_gibberish(question):
            response = "Hmm, aku kurang paham maksudmu. Bisa diulangi dengan lebih jelas?"
            context.update("gibberish", question, response)
            return AIResponse(response=response, session_id=session_id)
        
        category, rag_context = categorize_question_with_rag(question, context)
        
        if rag_context:
            mock_response = f"Berdasarkan informasi yang aku punya, {rag_context[:200]}... Apa yang ingin kamu tahu lebih detail?"
        else:
            mock_response = generate_fallback_response(question, category)
        
        related_topics = []
        if rag_system:
            try:
                related_topics = rag_system.suggest_related_topics(question)
            except Exception as e:
                logger.error(f"error getting related topics in mock: {e}")
        
        context.update(category, question, mock_response, rag_context)
        
        return AIResponse(
            response=mock_response,
            session_id=session_id, 
            related_topics=related_topics
        )
        
    except Exception as e:
        logger.error(f"error in mock endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

@app.get("/")
async def root():
    global rag_system, knowledge_base_data
    rag_status = "initialized" if rag_system else "fallback_mode"
    knowledge_count = len(knowledge_base_data) if knowledge_base_data else 0
    
    return {
        "message": "AI Portfolio Backend with Hybrid RAG is running", 
        "rag_system_type": RAG_SYSTEM_TYPE,
        "rag_status": rag_status,
        "knowledge_documents": knowledge_count,
        "features": {
            "chromadb_available": RAG_SYSTEM_TYPE == "ChromaDB",
            "simple_rag_available": RAG_SYSTEM_TYPE == "Simple", 
            "fallback_responses": RAG_SYSTEM_TYPE == "None"
        },
        "endpoints": ["/ask", "/ask-mock", "/suggested-followups/{session_id}", "/rag-status"]
    }

@app.get("/suggested-followups/{session_id}")
async def get_suggested_followups(session_id: str):
    if session_id not in conversation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    context = conversation_sessions[session_id]
    suggested_followups = context.get_suggested_followup_questions(3)
    
    return {"suggested_followups": suggested_followups}

@app.get("/rag-status")
async def get_rag_status():
    global rag_system, knowledge_base_data
    
    if rag_system:
        try:
            test_docs = rag_system.retrieve_relevant_docs("python", top_k=1)
            return {
                "status": "healthy",
                "system_type": RAG_SYSTEM_TYPE,
                "documents_indexed": len(knowledge_base_data),
                "test_query_results": len(test_docs),
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "vector_db": RAG_SYSTEM_TYPE,
                "fallback_mode": False
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
            "system_type": RAG_SYSTEM_TYPE,
            "reason": "RAG system not available, using static responses",
            "documents_indexed": len(knowledge_base_data),
            "fallback_mode": True
        }

@app.get("/knowledge-stats")
async def get_knowledge_stats():
    global knowledge_base_data
    
    if not knowledge_base_data:
        return {"error": "No knowledge base loaded"}
    
    category_stats = {}
    for doc in knowledge_base_data:
        category = doc.get('category', 'unknown')
        if category not in category_stats:
            category_stats[category] = 0
        category_stats[category] += 1
    
    return {
        "total_documents": len(knowledge_base_data),
        "categories": category_stats,
        "sample_titles": [doc.get('title', 'No title') for doc in knowledge_base_data[:5]],
        "rag_system": RAG_SYSTEM_TYPE
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)