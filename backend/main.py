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

app = FastAPI(title="AI Portfolio Backend")

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
        self.topic_transitions: List[Tuple[str, str, float]] = []  # (from_category, to_category, timestamp)
        self.last_response_category: Optional[str] = None
    
    def update(self, category: str, question: str, response: str = None):
        # deteksi transisi topik
        if self.last_category and self.last_category != category:
            self.topic_transitions.append((self.last_category, category, time.time()))
            # hapus potential followups jika topik berubah
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
                
            # hanya generate followups jika kategori sama atau related
            if not self.topic_transitions or self.topic_transitions[-1][1] == category:
                self.generate_potential_followups(category, question, response)
        
        self.detect_conversation_tone(question)
        
        for item in self.mentioned_items:
            if re.search(r'\b' + re.escape(item) + r'\b', question.lower()):
                self.referenced_items[item] += 1
        
        self.last_updated = time.time()
    
    def is_context_relevant(self, new_category: str) -> bool:
        """cek apakah konteks sebelumnya relevan untuk pertanyaan baru"""
        if not self.last_category:
            return False
        
        # jika kategori sama persis
        if new_category == self.last_category:
            return True
        
        # category groups yang related
        related_categories = {
            'keahlian': ['teknologi', 'proyek', 'tools'],
            'teknologi': ['keahlian', 'proyek'],
            'proyek': ['keahlian', 'teknologi'], 
            'makanan_favorit': [],  # makanan standalone
            'lagu_favorit': [],     # musik standalone
            'hobi': [],             # hobi standalone
        }
        
        # cek apakah kategori terkait
        if self.last_category in related_categories:
            if new_category in related_categories[self.last_category]:
                return True
        
        # cek followup patterns
        if new_category.endswith('_followup'):
            base_category = new_category.replace('_followup', '')
            return base_category == self.last_category
        
        # jika beda topik dan bukan followup, konteks tidak relevan
        return False
    
    def should_use_context_in_prompt(self, new_category: str) -> bool:
        """tentukan apakah konteks harus dimasukkan dalam prompt"""
        if not self.is_context_relevant(new_category):
            return False
        
        # jangan gunakan konteks jika sudah terlalu banyak transisi
        if len(self.topic_transitions) > 3:
            return False
        
        # jangan gunakan konteks jika transisi terakhir sudah lama
        if self.topic_transitions and (time.time() - self.topic_transitions[-1][2]) > 300:  # 5 menit
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
        food_patterns = [
            r'\b(martabak|sate|gorengan|ketoprak|batagor|nasi padang|soto|bakso|keripik|cimol|basreng)\b',
            r'street food (\w+)',
            r'makanan (\w+)'
        ]
        
        music_patterns = [
            r'\b(without you|air supply|glenn fredly|sekali ini saja|noah|separuh aku|celine dion|my heart will go on)\b',
            r'lagu "([^"]+)"',
            r'lagu ([A-Za-z\s]+) dari',
            r'\b(lo-fi|instrumental|soundtrack)\b'
        ]
        
        hobby_patterns = [
            r'\b(membaca|novel|omniscient reader|street food|traveling|wisata|destinasi)\b',
            r'hobi ([a-zA-Z\s]+)'
        ]
        
        tech_patterns = [
            r'\b(python|next\.js|react|data science|java|tailwind|typescript|fastapi|machine learning|pandas)\b',
            r'bahasa (\w+)',
            r'framework (\w+)'
        ]
        
        all_patterns = food_patterns + music_patterns + hobby_patterns + tech_patterns
        
        for pattern in all_patterns:
            matches = re.findall(pattern, response.lower())
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        for item in match:
                            if item and len(item) > 3:
                                self.mentioned_items.add(item.strip())
                    elif match and len(match) > 3:
                        self.mentioned_items.add(match.strip())
                        
    def generate_potential_followups(self, category: str, question: str, response: str):
        followups = {
            "makanan_favorit": [
                "Kenapa kamu suka {item}?",
                "Di mana tempat {item} favoritmu?",
                "Seberapa sering kamu makan {item}?",
                "Apa yang kamu suka dari {item}?",
                "{item} itu seperti apa sih?"
            ],
            "lagu_favorit": [
                "Kenapa kamu suka lagu {item}?",
                "Kapan pertama kali kamu dengar {item}?",
                "Apa lirik favoritmu dari {item}?",
                "Ada rekomendasi lagu mirip {item}?",
                "Apa lagu lain dari {artist} yang kamu suka?"
            ],
            "hobi": [
                "Sejak kapan kamu suka {item}?",
                "Berapa sering kamu {item}?",
                "Apa yang menarik dari {item}?",
                "Bisa cerita pengalaman menarik saat {item}?",
                "Apa tantangan dalam {item}?"
            ],
            "keahlian": [
                "Bagaimana kamu belajar {item}?",
                "Apa proyek {item} yang pernah kamu buat?",
                "Kenapa kamu tertarik dengan {item}?",
                "Berapa lama kamu sudah menguasai {item}?",
                "Tools apa yang kamu pakai untuk {item}?"
            ],
            "proyek": [
                "Apa tantangan terbesar dalam membuat {item}?",
                "Teknologi apa yang kamu pakai untuk {item}?",
                "Berapa lama kamu mengerjakan {item}?",
                "Apa yang kamu pelajari dari {item}?",
                "Bagaimana feedback orang lain tentang {item}?"
            ]
        }
        
        items_in_response = []
        for item in self.mentioned_items:
            if item in response.lower():
                items_in_response.append(item)
                
        if category in followups and items_in_response:
            potential_questions = []
            for item in items_in_response:
                for template in followups[category]:
                    try:
                        if '{item}' in template:
                            potential_questions.append(template.format(item=item))
                        elif '{artist}' in template and category == "lagu_favorit":
                            artist_match = re.search(r'(.+) dari (.+)', response.lower())
                            if artist_match:
                                artist = artist_match.group(2).strip()
                                potential_questions.append(template.format(artist=artist))
                    except:
                        continue
                        
            if potential_questions:
                self.potential_followups[category] = potential_questions[:5]

    def get_most_referenced_items(self, limit=3):
        return sorted(self.referenced_items.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def get_suggested_followup_questions(self, limit=3):
        if not self.last_category or self.last_category not in self.potential_followups:
            return []
            
        return self.potential_followups[self.last_category][:limit]

conversation_sessions = {}

def cleanup_old_sessions():
    current_time = time.time()
    sessions_to_remove = []
    
    for session_id, context in conversation_sessions.items():
        if current_time - context.last_updated > 1800:  # 30 menit
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del conversation_sessions[session_id]
    
    if sessions_to_remove:
        logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")

# data profil pengguna
user_profile = {
    "nama": "Danendra Shafi Athallah",
    "lokasi": "Jakarta, Indonesia",
    "pendidikan": "Institut Teknologi Bandung, Teknik Informatika (Semester 4)",
    "pendidikan_sebelumnya": {
        "sd": "SD Islam Al Azhar 23 Jatikramat",
        "smp": "SMP Islam Al Azhar 9 Kemang Pratama",
        "sma": "SMA Negeri 5 Bekasi",
    },
    "pekerjaan": "Mahasiswa",
    "pengalaman": "2 tahun pengalaman di pengembangan web dan 1 tahun di data science",
    "keahlian": ["Next.js", "React", "Python", "Data Science", "Java", "Tailwind CSS"],
    "keahlian_detail": {
        "Next.js": "Framework utama yang digunakan untuk berbagai proyek web selama 2 tahun terakhir",
        "React": "Library JavaScript favorit untuk membangun UI yang interaktif",
        "Python": "Bahasa pemrograman utama untuk analisis data dan pengembangan algoritma",
        "Data Science": "Analisis data menggunakan pandas, matplotlib, dan scikit-learn",
        "Java": "Bahasa pemrograman untuk pengembangan algoritma dan aplikasi desktop"
    },
    "tools_favorit": {
        "Python": "Bahasa utama untuk analisis data dan machine learning",
        "VS Code": "Editor favorit dengan banyak extension untuk produktivitas",
        "Jupyter Notebook": "Untuk eksplorasi dan visualisasi data",
        "Git": "Version control untuk kolaborasi dan tracking proyek",
        "Figma": "Untuk wireframing dan design",
    },
    "hobi": ["Membaca buku novel", "Traveling ke destinasi lokal", "Penggemar street food"],
    "hobi_detail": {
        "Membaca": "Buku favorit termasuk 'Omniscient Reader Viewpoint' dan novel fantasi",
        "Traveling": "Sudah mengunjungi 4 provinsi di Indonesia dan berencana menambah lagi",
        "Street Food": "Penggemar berat street food Indonesia seperti martabak, sate, dan gorengan"
    },
    "makanan_favorit": {
        "street_food": ["Martabak manis", "Sate ayam", "Gorengan", "Ketoprak", "Batagor"],
        "restoran": ["Nasi padang", "Soto betawi", "Bakso"],
        "camilan": ["Keripik pisang", "Basreng", "Cimol"]
    },
    "proyek": [
        "Algoritma Pencarian Little Alchemy 2 - Implementasi BFS, DFS, dan Bidirectional Search",
        "Rush Hour Puzzle Solver - Program penyelesaian puzzle dengan algoritma pathfinding",
        "Personal Finance Tracker - Aplikasi tracking keuangan pribadi",
        "IQ Puzzler Pro Solver - Solusi permainan papan menggunakan algoritma brute force"
    ],
    "proyek_detail": {
        "Algoritma Pencarian Little Alchemy 2": "Implementasi BFS, DFS, dan Bidirectional Search untuk mencari kombinasi recipe dalam permainan. Seru banget menyelesaikan tantangan algoritma ini!",
        "Rush Hour Puzzle Solver": "Program yang menyelesaikan puzzle Rush Hour menggunakan algoritma pathfinding seperti UCS, Greedy Best-First Search, A*, dan Dijkstra. Dilengkapi dengan CLI dan GUI untuk visualisasi solusi. Salah satu proyek yang paling menantang dari segi algoritma.",
        "Personal Finance Tracker": "Aplikasi web yang membantu pengguna melacak pengeluaran, mengatur anggaran, dan memvisualisasikan kebiasaan finansial. Menggunakan React, Firebase, dan D3.js untuk visualisasi data yang interaktif.",
        "IQ Puzzler Pro Solver": "Solusi untuk permainan papan IQ Puzzler Pro menggunakan algoritma brute force dengan visualisasi interaktif. Butuh optimasi yang cukup rumit biar performanya bagus."
    },
    "tantangan_proyek": {
        "Algoritma Pencarian Little Alchemy 2": "Tantangan terbesar adalah memaksimalkan efisiensi algoritma untuk pencarian kombinasi recipe yang banyak. Bidirectional search dibuat untuk mengatasi bottleneck pada graf hubungan recipe yang kompleks.",
        "Rush Hour Puzzle Solver": "Optimalisasi algoritma A* dengan heuristik custom agar performa lebih baik. Tantangan lain adalah visualisasi state puzzle yang interaktif dengan library grafis yang terbatas.",
        "IQ Puzzler Pro Solver": "Tantangan utama adalah state space yang sangat besar, karena banyaknya kombinasi yang mungkin. Perlu implementasi backtracking dengan pruning yang efisien untuk mencegah stack overflow."
    },
    "karakter": "Kreatif, analitis, detail-oriented, dan suka belajar hal baru",
    "sifat_detail": {
        "keberanian": "Mudah beradaptasi di lingkungan baru dan berani mengambil tantangan",
        "kerjasama": "Bisa menyesuaikan peran sebagai pemimpin atau anggota tim sesuai kebutuhan",
        "komunikasi": "Terbuka dalam komunikasi, senang berdiskusi tentang ide dan konsep baru"
    },
    "prestasi": [
        "Asisten praktikum Berpikir Komputasional"
    ],
    "lomba": {
        "Datathon UI": "Pengalaman lomba data science yang paling berkesan karena kompleksitasnya yang menantang"
    },
    "quotes_favorit": [
        "Code is like humor. When you have to explain it, it's bad.",
        "The best way to predict the future is to create it.",
        "Simplicity is the ultimate sophistication."
    ],
    "moto": "Menuju tak terbatas dan melampauinya",
    "lagu_favorit": {
        "Indonesia": ["Sekali Ini Saja - Glenn Fredly", "Separuh Aku - Noah", "Cinta sudah lewat - Kahitna"],
        "Barat": ["Without You - Air Supply", "My Heart Will Go On - Celine Dion", "Perfect - Ed Sheeran"],
        "Genre": ["Pop 90an", "Ballad", "Oldies"],
        "Untuk_Coding": ["Lagu instrumental", "Lo-fi beats", "Soundtrack film"]
    },
    "kuliah": {
        "mata_kuliah_favorit": "Matematika",
        "pengalaman_culture_shock": "Kuliah di ITB memberikan culture shock karena banyak mahasiswa sudah fasih dengan dunia IT sejak kecil, berbeda dengan saya yang baru memulai. Pace pembelajaran yang sangat cepat juga membuat saya harus beradaptasi dengan baik.",
        "organisasi": "Kepanitiaan Arkavidia divisi academy untuk bootcamp path data science"
    },
    "belajar_coding": {
        "pertama_kali": "SMA",
        "data_science": "Mulai belajar data science dari Excel dan visualisasi data sederhana"
    },
    "manajemen": {
        "waktu": "Membagi waktu antara mengerjakan projek, tugas besar, dan belajar untuk ujian dengan sangat ketat",
        "stres": "Menonton film horror/romance atau drama Korea untuk relaksasi",
        "bekerja_tim": "Melihat dulu apakah ada yang mau menginisiasi menjadi leader, kalau tidak ada baru saya ambil peran tersebut"
    },
    "personality": {
        "tipe": "Mudah berkenalan dengan orang baru",
        "kebiasaan_ngoding": "Terkadang lebih produktif saat ngoding malam hari"
    },
    "rencana_masa_depan": "Fokus memperdalam keahlian di bidang data science dan algoritma, lulus dengan prestasi terbaik, dan berkarir di perusahaan teknologi terkemuka.",
    "portfolio_tech": {
        "frontend": "Next.js, TypeScript, Tailwind CSS, Shadcn UI, Framer Motion",
        "backend": "Python FastAPI, OpenAI API",
        "deployment": "Vercel untuk frontend, Railway untuk backend Python",
        "design": "Menggunakan prinsip mobile-first design dengan animasi smooth dan interaksi intuitif"
    }
}

def is_gibberish(text: str) -> bool:
    if len(text) < 2:
        return False
        
    text = text.lower()
    
    # pertanyaan singkat tentang teknologi diterima
    if any(tech in text for tech in ["python", "react", "java", "next.js", "data science"]):
        return False
    
    # pertanyaan dengan "ceritakan" atau "bagaimana" diterima
    if any(key in text for key in ["ceritakan", "bagaimana", "gimana", "tentang"]):
        return False
    
    # cek jika pertanyaan terlalu pendek (< 2 kata)
    if len(text.split()) < 2:
        # hanya valid jika ini adalah kata kunci khusus
        valid_terms = ["python", "react", "java", "next", "hobi", "makanan", "lagu", "proyek"]
        return not any(term in text for term in valid_terms)
    
    # deteksi konsonan berturut-turut yang tidak wajar
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
    
    # hitung rasio vokal
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

def detect_question_intent(question: str, context: ConversationContext = None) -> str:
    question_lower = question.lower()
    
    # challenging questions
    challenging_patterns = [
        r'kenapa (?:saya|aku) harus',
        r'apa (?:yang )?membuat (?:kamu|anda)',
        r'(?:emang|memang|masa) (?:cukup|bisa)',
        r'(?:yakin|serius) (?:bisa|mampu)',
        r'hanya|cuma|saja|doang',
        r'(?:gak|tidak|nggak) (?:cukup|bisa)',
    ]
    
    # reflective questions
    reflective_patterns = [
        r'(?:gimana|bagaimana) (?:menurutmu|pendapatmu)',
        r'kamu (?:rasa|pikir|anggap)',
        r'menurut (?:kamu|anda)',
        r'apa (?:pendapat|opini)',
    ]
    
    # comparison questions
    comparison_patterns = [
        r'(?:dibanding|dibandingkan)',
        r'lebih (?:baik|bagus)',
        r'versus|vs',
        r'mana yang',
    ]
    
    # follow up questions
    followup_patterns = [
        r'(?:terus|lalu|kemudian) (?:gimana|bagaimana)',
        r'(?:selain|kecuali) (?:itu|dari)',
        r'balik ke',
        r'kembali ke',
        r'(?:emang|memang) (?:cukup|bisa)',
        r'(?:yakin|serius) (?:gak|tidak)',
        r'hanya (?:dengan|pakai)',
        r'cuma (?:dengan|pakai)',
    ]
    
    for pattern in challenging_patterns:
        if re.search(pattern, question_lower):
            return "challenging"
    
    for pattern in reflective_patterns:
        if re.search(pattern, question_lower):
            return "reflective"
    
    for pattern in comparison_patterns:
        if re.search(pattern, question_lower):
            return "comparison"
    
    # cek followup hanya jika ada konteks yang relevan
    if context and context.is_context_relevant("followup"):
        for pattern in followup_patterns:
            if re.search(pattern, question_lower):
                return "followup"
    
    return "neutral"

def detect_specific_question_type(question: str, category: str) -> str:
    """deteksi tipe pertanyaan spesifik dalam kategori"""
    question_lower = question.lower()
    
    if category == "lagu_favorit":
        if any(pattern in question_lower for pattern in ["kapan pertama", "pertama kali", "pertama dengar"]):
            return "first_time_hearing"
        elif any(pattern in question_lower for pattern in ["kenapa suka", "mengapa suka", "apa yang suka"]):
            return "why_like"
        elif any(pattern in question_lower for pattern in ["lirik favorit", "lirik kesukaan", "bagian favorit"]):
            return "favorite_lyrics"
        elif any(pattern in question_lower for pattern in ["rekomendasi", "recommend", "lagu lain", "mirip"]):
            return "recommendation"
        elif any(pattern in question_lower for pattern in ["arti", "makna", "bermakna"]):
            return "meaning"
    
    elif category == "makanan_favorit":
        if any(pattern in question_lower for pattern in ["kenapa suka", "mengapa suka", "apa yang suka"]):
            return "why_like"
        elif any(pattern in question_lower for pattern in ["di mana", "dimana", "tempat", "lokasi"]):
            return "where_to_eat"
        elif any(pattern in question_lower for pattern in ["seberapa sering", "berapa sering", "kapan makan"]):
            return "frequency"
        elif any(pattern in question_lower for pattern in ["rekomendasi", "recommend", "suggest"]):
            return "recommendation"
    
    elif category == "hobi":
        if any(pattern in question_lower for pattern in ["sejak kapan", "kapan mulai", "mulai kapan"]):
            return "when_started"
        elif any(pattern in question_lower for pattern in ["berapa sering", "seberapa sering"]):
            return "frequency"
        elif any(pattern in question_lower for pattern in ["apa menarik", "yang menarik", "kenapa suka"]):
            return "what_interesting"
        elif any(pattern in question_lower for pattern in ["pengalaman", "cerita", "momen"]):
            return "experience_story"
    
    return "general"

# categories keywords untuk deteksi yang lebih nuanced
category_keywords = {
    "lagu_favorit": [
        "lagu", "musik", "dengerin", "dengarkan", "nyanyi", "penyanyi", "band", "playlist",
        "genre", "album", "jadul", "konser", "artist", "artis", "musisi", "spotify", "instrumental", 
        "mendengarkan", "lagu favorit", "lagu kesukaan", "musik favorit", "enak didengerin",
        "air supply", "glenn fredly", "without you", "sekali ini saja", "celine dion", 
        "oldies", "lo-fi", "soundtrack", "pop", "kapan pertama", "pertama kali dengar",
        "kenapa suka", "lirik favorit", "rekomendasi lagu"
    ],
    "makanan_favorit": [
        "makanan", "makan", "masak", "sarapan", "makan siang", "makan malam", "kuliner", 
        "masakan", "jajan", "ngemil", "makanan favorit", "masakan favorit", "kuliner favorit",
        "street food", "makanan jalanan", "enak", "lezat", "gurih", "manis", "pedas", "snack",
        "hidangan", "menu", "lapar", "kenyang", "nyemil", "nongkrong", "martabak", "sate",
        "gorengan", "ketoprak", "batagor", "nasi padang", "soto", "bakso", "keripik", "camilan"
    ],
    "hobi": [
        "hobi", "suka", "waktu luang", "kegiatan", "aktivitas", "senang", "menghabiskan waktu",
        "passion", "interest", "mengisi waktu", "kesenangan", "leisure", "weekend", "libur",
        "membaca", "buku", "novel", "traveling", "jalan-jalan", "wisata", "destinasi", 
        "omniscient reader", "fantasi", "sci-fi"
    ],
    "keahlian": [
        "keahlian", "skill", "kemampuan", "ahli", "bisa apa", "bisa apa saja", "jago", 
        "expert", "menguasai", "expertise", "kompetensi", "kapabilitas", "spesialisasi", 
        "teknis", "technical", "hard skill", "python", "javascript", "react", "next.js",
        "data science", "programming", "coding"
    ],
    "proyek": [
        "proyek", "project", "karya", "portfolio", "aplikasi", "buat apa", "telah dibuat", 
        "terbaik", "unggulan", "kerjaan", "hasil", "pencapaian", "showcase", "demo", "showcase",
        "alchemy", "rush hour", "puzzle", "algoritma", "finance tracker"
    ],
    "rekrutmen": [
        "rekrut", "hire", "merekrut", "tim", "team", "kerja sama", "bergabung",
        "harus merekrut", "kenapa harus", "mengapa memilih", "apa yang membuat"
    ]
}

def preprocess_question(question: str) -> str:
    question = question.lower()
    question = re.sub(r'[^\w\s]', ' ', question)
    question = re.sub(r'\s+', ' ', question).strip()
    return question

def categorize_question(question: str, context: ConversationContext = None) -> str:
    original_question = question
    question = preprocess_question(question)
    question_words = question.split()
    
    logger.info(f"Processing question: {question}")
    
    # deteksi gibberish
    if is_gibberish(question):
        logger.info(f"Detected gibberish: {question}")
        return "gibberish"
    
    # pertanyaan sangat singkat tentang teknologi
    if len(question_words) <= 2:
        tech_keywords = ["python", "react", "java", "next.js", "data science"]
        if any(keyword in question.lower() for keyword in tech_keywords):
            return "teknologi"
    
    # deteksi pertanyaan khusus rekrutmen/hiring
    recruitment_patterns = [
        r'kenapa (?:saya|aku) harus (?:merekrut|hire)',
        r'apa (?:yang )?membuat (?:kamu|anda) (?:cocok|layak)',
        r'mengapa (?:memilih|pilih) (?:kamu|anda)',
        r'(?:keunggulan|kelebihan) (?:kamu|anda)',
        r'tim (?:data|ml|machine learning)',
        r'(?:hire|rekrut) (?:kamu|anda)'
    ]
    
    for pattern in recruitment_patterns:
        if re.search(pattern, question):
            return "rekrutmen"
    
    if context:
        # cek apakah ini followup berdasarkan pattern dan relevansi konteks
        question_intent = detect_question_intent(question, context)
        if question_intent == "followup" and context.is_context_relevant("followup"):
            # tambahkan suffix followup hanya jika benar-benar followup
            return context.last_category + "_followup" if context.last_category else "general"
    
    # pengecekan kategori personal
    personal_checks = [
        (["pacar", "jodoh", "pacaran", "pasangan", "gebetan", "wanita", "nikah", "menikah", "single", "lajang", "status hubungan"], "personal_relationship"),
        (["gaji", "salary", "penghasilan", "bayaran", "uang", "kekayaan", "sebulan", "pendapatan"], "personal_financial"),
        (["alamat rumah", "tinggal dimana", "alamat lengkap", "nomor", "kontak", "pribadi", "telepon", "hp"], "personal_contact"),
        (["umur", "usia", "tanggal lahir", "kapan lahir", "kelahiran", "berapa tahun"], "personal_age"),
        (["agama", "kepercayaan", "tuhan", "beribadah", "keyakinan", "ibadah"], "personal_religion")
    ]
    
    for keywords, category in personal_checks:
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', question) for keyword in keywords):
            return category
    
    # cek untuk kategori spesifik
    word_category_counts = {}
    
    for category, keywords in category_keywords.items():
        matches = 0
        matched_words = []
        
        # cek kata-kata eksak
        for word in question_words:
            if any(re.search(r'\b' + re.escape(word) + r'\b', keyword) or 
                   re.search(r'\b' + re.escape(keyword) + r'\b', word) 
                   for keyword in keywords):
                matches += 1
                matched_words.append(word)
        
        # cek frasa lengkap
        for keyword in keywords:
            if ' ' in keyword and re.search(r'\b' + re.escape(keyword) + r'\b', question):
                matches += 2
                matched_words.append(keyword)
        
        word_category_counts[category] = matches
    
    # tambahkan kategori lain
    other_categories = {
        "mata_kuliah": ["pelajaran favorit", "mata kuliah favorit", "mata pelajaran", "pelajaran", "kuliah favorit"],
        "lokasi": ["lokasi", "tinggal", "domisili", "alamat", "kota", "daerah", "rumah"],
        "prestasi": ["prestasi", "pencapaian", "award", "penghargaan", "juara"],
        "lomba": ["lomba", "kompetisi", "contest", "hackathon", "datathon"],
        "data_science": ["data", "data science", "analisis data", "big data", "statistik", "machine learning", "ml"],
        "tools": ["tool", "alat", "software", "library", "framework", "favorit", "suka pakai"],
        "karakter": ["karakter", "kepribadian", "sifat", "tipe", "mbti", "orangnya", "pemalu", "extrovert", "introvert"],
        "portofolio_tech": ["portofolio ini", "website ini", "web ini", "dibuat pakai", "teknologi"],
        "rencana": ["rencana", "masa depan", "target", "tujuan", "cita", "5 tahun"],
        "pekerjaan": ["pekerjaan", "kerja", "profesi", "karir", "jabatan"],
        "pengalaman": ["pengalaman", "experience", "lama kerja"],
        "manajemen_waktu": ["waktu", "manage", "manajemen", "atur waktu", "produktif"],
        "manajemen_stres": ["stres", "stress", "tekanan", "pressure", "beban", "handle"],
        "cerita_kuliah": ["cerita", "momen", "pengalaman kuliah", "culture shock", "berkesan"],
        "organisasi": ["organisasi", "berorganisasi", "komunitas", "kepanitiaan"],
        "belajar_mandiri": ["belajar mandiri", "autodidak", "self-taught", "tutorial"],
        "belajar_kegagalan": ["kegagalan", "gagal", "failure", "kesalahan", "mistake"],
        "kerja_tim": ["tim", "team", "kerja tim", "kolaborasi", "konflik"],
        "kebiasaan_ngoding": ["ngoding", "coding", "kode", "malam", "produktif"],
        "moto_hidup": ["moto", "motto", "quotes", "quote", "kutipan", "kata-kata"]
    }
    
    for category, keywords in other_categories.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', question):
                word_category_counts[category] = word_category_counts.get(category, 0) + 1
    
    # pemeriksaan khusus untuk pendidikan
    education_keywords = ["pendidikan", "sekolah", "kuliah", "belajar", "kampus", "universitas", "itb", "masuk itb", "masuk kuliah", "jurusan"]
    if any(re.search(r'\b' + re.escape(keyword) + r'\b', question) for keyword in education_keywords):
        return "pendidikan"
    
    # tentukan kategori dengan skor tertinggi
    if word_category_counts:
        top_category = max(word_category_counts.items(), key=lambda x: x[1])
        if top_category[1] > 0:
            logger.info(f"Selected category with score: {top_category}")
            return top_category[0]
    
    # cek apakah pertanyaan python (khusus karena sering digunakan)
    if "python" in question.lower():
        return "teknologi"
    
    # cek apakah pertanyaan terlalu pendek atau kurang jelas
    if len(question.split()) < 3:
        return "unclear_question"
    
    return "general"

def generate_interactive_response(question: str, category: str, context: ConversationContext = None) -> str:
    # deteksi intent untuk konteks, tapi jangan ekspos ke user
    question_intent = detect_question_intent(question, context)
    
    # deteksi tipe pertanyaan spesifik
    specific_type = detect_specific_question_type(question, category)
    logger.info(f"Specific question type: {specific_type}")
    
    # opener yang natural tanpa format array
    simple_openers = [
        "Soal itu,",
        "Hm,",
        "Kalau itu,",
        "",  # kadang langsung jawab tanpa opener
    ]
    
    # PERBAIKI: untuk follow-up questions, cek relevansi konteks dulu
    if question_intent == "followup" and context and context.is_context_relevant(category):
        followup_openers = [
            "Oh, balik ke topik tadi ya?",
            "Iya, soal itu tadi...",
            "Right,",
            ""
        ]
        opener = random.choice(followup_openers)
    else:
        opener = random.choice(simple_openers)
    
    # KATEGORI SERIUS - Response panjang dan detail
    if category == "rekrutmen":
        content_responses = [
            "Aku bawa kombinasi teori dan praktek di data science yang cukup solid. Udah terapin langsung di proyek kayak Rush Hour Puzzle Solver dengan implementasi algoritma pathfinding kompleks seperti UCS, A*, dan Dijkstra. Plus pengalaman dari berbagai lomba termasuk Datathon UI yang ngasah kemampuan handle dataset kompleks, dan jadi asisten praktikum yang ngelatih communication skills. Yang bikin aku beda adalah dedikasi buat terus belajar teknologi baru dan nggak takut hadapi tantangan yang belum pernah dicoba sebelumnya.",
            
            "Yang bikin aku menonjol adalah cara pendekatan masalah yang sistematis dan analitis. Dengan pengalaman 2 tahun web development dan 1 tahun fokus data science, aku udah develop mindset untuk breakdown complex problems jadi manageable pieces. Contohnya di proyek Little Alchemy Solver, aku harus implement BFS, DFS, dan Bidirectional Search untuk optimize recipe search di graph yang kompleks. Bukan cuma skill teknis yang solid, tapi juga kemampuan komunikasikan insight dan findings ke stakeholder dengan cara yang mudah dipahami. Plus track record akademik yang konsisten dan keterlibatan aktif di organisasi tech kayak Arkavidia.",
            
            "Track record aku di berbagai domain cukup membuktikan versatility dan konsistensi. Mulai dari academic projects yang challenging, participation di kompetisi data science, sampai kontribusi open source di proyek React. Yang paling menonjol adalah implementasi algoritma advanced di berbagai puzzle solver - dari state space optimization sampai heuristic design. Experience ini ngasih aku solid foundation dalam algorithm design dan performance tuning. Plus, background sebagai asisten praktikum dan active member di komunitas tech bikin aku comfortable dalam collaborative environment dan knowledge sharing.",
            
            "Passion aku di data science itu genuine dan bukan cuma trend following. Ini terbukti dari consistency dalam deliver hasil berkualitas, baik di tugas kuliah yang demanding, participation di lomba-lomba competitive, maupun contribution ke proyek open source. Yang jadi kekuatan utama adalah learning agility - aku cepat adapt dengan teknologi baru dan selalu penasaran sama challenge yang belum pernah encountered sebelumnya. Contohnya, dalam waktu singkat aku bisa master different algorithm paradigms dan apply mereka ke real-world problems dengan effective optimization strategies.",
            
            "Aku nggak cuma bawa hard skills yang solid, tapi juga fresh perspective dari dunia akademik dan exposure ke latest research trends. Kombinasi theoretical knowledge di algorithm design dengan hands-on experience di practical implementation bikin aku bisa bridge gap antara research dan application. Skill set aku cover dari data preprocessing dengan pandas, modeling dengan scikit-learn, sampai full-stack development dengan Next.js. Yang penting, aku selalu fokus deliver actionable insights dan maintainable solutions, bukan cuma proof of concept yang impressive tapi susah di-scale.",
            
            "Aku tipe yang thrive dalam collaborative environment dan genuine enjoy sharing knowledge dengan team members. Experience sebagai asisten praktikum dan involvement di organizing committees ngajarin aku gimana cara effective communication dengan diverse backgrounds. Technical skills aku solid - proven di berbagai algorithmic projects dan consistent academic performance - tapi yang lebih valuable adalah ability untuk facilitate knowledge transfer dan mentor junior team members. Plus, aku always open untuk feedback dan continuous improvement, yang crucial dalam fast-paced tech environment.",
            
            "Kombinasi unik aku adalah strong academic foundation yang dibalance dengan practical project experience dan soft skills development. Aku nggak cuma understand algorithms secara theoretical, tapi udah implement dan optimize mereka untuk real-world constraints. Portfolio projects kayak IQ Puzzler Pro Solver dan Rush Hour Puzzle menunjukkan kemampuan handle complex state spaces dan develop efficient solutions. Plus, active involvement di tech community dan teaching experience bikin aku comfortable dengan knowledge dissemination dan cross-functional collaboration.",
            
            "Kelebihan utama aku adalah systematic approach dalam problem decomposition dan solution architecture. Terbukti dari success rate di berbagai challenging projects yang require both algorithmic thinking dan engineering practicality. Aku bisa efficiently analyze requirements, design optimal approaches, dan implement robust solutions yang scalable. Experience covering different domains - dari graph algorithms sampai web development - kasih aku versatile skill set yang applicable ke various business problems. Plus mindset yang always question assumptions dan explore alternative approaches untuk ensure optimal outcomes."
        ]
    
    elif category.endswith("_followup"):
        base_category = category.replace("_followup", "")
        
        if context and context.is_context_relevant(category):
            if question_intent == "challenging":
                content_responses = [
                    "'Cukup' itu memang relatif, tapi aku confident dengan depth dari pengalaman yang udah dijalani. Dalam 2-3 tahun terakhir, aku nggak cuma belajar surface level tapi bener-bener deep dive ke setiap technology stack dan methodology. Contohnya di Rush Hour Solver, aku spend significant time untuk optimize algorithm performance dari naive approach sampai sophisticated heuristic design. Setiap project ngajarin unique challenges - dari memory management di large state spaces sampai UI/UX considerations untuk complex visualizations. Plus, academic environment di ITB constantly expose aku ke cutting-edge research dan best practices yang immediately applicable ke practical problems.",
                    
                    "Fair point untuk questioning that. Learning curve di tech industry memang steep dan landscape-nya terus berubah rapidly. Tapi yang aku notice adalah foundation yang solid di fundamentals bikin adaptation ke new technologies jadi much smoother. Pengalaman sekarang udah cover diverse areas - algorithm design, data analysis, web development, dan system optimization. Yang penting adalah aku develop strong problem-solving methodology yang transferable across domains. Plus, habit untuk continuous learning dan engagement dengan tech community ensure aku stay updated dengan industry trends dan emerging technologies.",
                    
                    "Honestly, aku acknowledge bahwa experience aku belum se-extensive senior developers dengan decades of industry background. Tapi what I lack in years, aku compensate dengan intensity dan breadth of learning. Every single project aku treat sebagai comprehensive learning opportunity - from initial research phase sampai post-implementation analysis. Academic setting di ITB juga provide unique advantages seperti access ke latest research, structured approach to problem-solving, dan opportunity untuk experiment dengan novel approaches without commercial pressure constraints."
                ]
            else:
                # followup yang tidak challenging - berdasarkan specific type
                if base_category == "lagu_favorit":
                    if specific_type == "first_time_hearing":
                        content_responses = [
                            "Pertama kali dengar Glenn Fredly waktu SMA, pas lagi ada acara keluarga dan aku putar 'Sekali Ini Saja'. Voice dia yang khas dan lirik yang deep banget langsung bikin aku terpukau. Dari situ mulai explore album-albumnya yang lain. Gaya musiknya yang soulful tapi tetap Indonesian banget bikin aku appreciate local music lebih dalam.",
                            
                            "Awal kenal Glenn Fredly pas lagi nyari musik untuk playlist study. Temen recommend 'Akhir Cerita Cinta' dan ternyata cocok banget buat background music coding. Setelah itu jadi curious sama karya-karya lainnya. Yang bikin special adalah combine antara jazz, soul, dan nuansa Indonesia yang unique."
                        ]
                    elif specific_type == "why_like":
                        content_responses = [
                            "Glenn Fredly special karena voice quality dan songwriting ability yang luar biasa. Lagu-lagunya punya depth yang nggak cuma catchy, tapi meaningful. Plus dia pioneer dalam Indonesian jazz-soul musik, yang inspiring banget buat industry musik kita.",
                            
                            "Yang bikin aku suka Glenn Fredly adalah versatility dalam musicality. Dari ballad romantis sampai jazz-funk, semua dia handle dengan sempurna. Lirik-liriknya juga puitis, nggak asal rhyme tapi benar-benar bercerita."
                        ]
                    elif specific_type == "recommendation":
                        content_responses = [
                            "Kalau suka Glenn Fredly, coba dengar Andien atau Indra Lesmana. Music style mereka punya similarity dalam jazz-soul Indonesian. RAN juga bagus, especially untuk modern Indonesian pop yang high quality.",
                            
                            "Untuk vibe yang mirip Glenn Fredly, recommend banget Tompi, Sade Merah Putih, atau even some songs dari Raisa yang lebih jazz-oriented. Atau kalau mau explore international, coba John Mayer atau D'Angelo."
                        ]
                    else:
                        content_responses = [
                            "Glenn Fredly memang legend di Indonesian music scene. Karya-karyanya timeless dan terus relevan sampai sekarang.",
                            "Musik Glenn Fredly punya emotional connection yang strong. Setiap dengerin selalu ada moment yang relate."
                        ]
                elif base_category == "keahlian":
                    content_responses = [
                        "Pengalaman aku di data science specifically focus pada end-to-end implementation yang solve real problems. Mulai dari data acquisition dan preprocessing, exploratory analysis untuk understand patterns, feature engineering untuk optimize model performance, sampai deployment considerations untuk production environment. Portfolio projects demonstrate kemampuan handle different types of challenges - optimization problems di puzzle solvers, predictive modeling di finance tracker, dan algorithm visualization untuk educational purposes.",
                        
                        "Skill development journey aku quite structured dan progressive. Dimulai dari solid foundation di programming fundamentals, then expanding ke specialized areas seperti machine learning, algorithm optimization, dan web development. Each phase build upon previous knowledge - web dev experience help dengan deployment considerations, algorithm background crucial untuk model optimization, dan data analysis skills essential untuk business insight generation. Current focus adalah integrating AI capabilities dengan practical applications, seperti yang demonstrated di portfolio ini."
                    ]
                elif base_category == "proyek":
                    content_responses = [
                        "Detail implementasi Rush Hour Solver specifically involve custom heuristic design yang account untuk both distance to goal dan board configuration complexity. Algorithm comparison menunjukkan A* dengan Manhattan distance heuristic provide optimal balance antara solution quality dan computation time. Plus, visualization component menggunakan JavaFX yang allow real-time algorithm execution tracking with smooth animation transitions.",
                        
                        "Little Alchemy project technically challenging karena require bidirectional search implementation dalam dynamic recipe graph. Optimization include memoization untuk prevent redundant path calculations dan pruning strategies untuk reduce search space. Performance analysis menunjukkan bidirectional approach reduce average search time by 60% compared dengan standard BFS approach, especially untuk complex recipe dependencies."
                    ]
                else:
                    # default followup responses
                    content_responses = [
                        f"Ya, soal {base_category} tadi, aku memang passionate tentang hal tersebut. Ada specific aspect yang ingin kamu tahu lebih detail?",
                        f"Terkait {base_category}, pengalaman aku cukup beragam. Apa yang mau digali lebih dalam?",
                        f"Oh iya, {base_category}. Ada curiosity khusus tentang hal itu?"
                    ]
        else:
            # jika konteks tidak relevan, treat as new question
            category = base_category
            content_responses = [
                f"Hmm, sepertinya topik berpindah ke {base_category} ya. Aku senang bahas hal ini.",
                f"Oh, sekarang tentang {base_category}. Topik yang menarik nih.",
                f"Baik, kita bahas {base_category} sekarang."
            ]
    
    elif category == "keahlian":
        content_responses = [
            "Skill utama aku ada di intersection antara data science dan web development, dengan strong foundation di Python dan JavaScript ecosystems. Untuk data science, aku proficient dengan pandas untuk data manipulation, scikit-learn untuk machine learning models, matplotlib dan seaborn untuk visualization, dan numpy untuk numerical computing. Web development side, aku experienced dengan Next.js dan React untuk frontend, plus kemampuan integrate dengan backend services. Yang crucial adalah understanding tentang when to use which tool - efficiency dan hasil akhir selalu jadi primary consideration dalam technology selection.",
            
            "Specialization aku adalah combining data science capabilities dengan modern web development, yang relatively unique combination. Python jadi daily driver untuk all data-related tasks - from preprocessing messy datasets sampai building predictive models. Next.js dan React ecosystem aku gunakan untuk create interactive applications yang showcase data insights effectively. Portfolio ini perfect example dari integration tersebut - AI backend dengan FastAPI, modern frontend dengan TypeScript, dan seamless user experience. Plus, aku comfortable dengan deployment strategies untuk both data science models dan web applications.",
            
            "Yang bikin aku confident adalah extensive hands-on experience dengan tools yang aku claim sebagai expertise. Pandas bukan cuma untuk basic data manipulation, tapi advanced operations seperti multi-index handling, time series analysis, dan performance optimization untuk large datasets. Scikit-learn usage cover dari classical algorithms sampai ensemble methods, dengan understanding tentang hyperparameter tuning dan cross-validation strategies. React development includes state management dengan complex component architectures, performance optimization dengan memoization, dan integration dengan various APIs dan databases.",
            
            "Approach aku dalam skill development adalah depth over breadth, dengan focus ke technologies yang proven valuable dan versatile. Python ecosystem dipilih karena extensive libraries untuk data science, machine learning, dan general programming. JavaScript dengan React/Next.js chosen karena powerful untuk building modern, interactive applications. But aku always open untuk explore new technologies kalau memang solve specific problems better. Recent exploration includes FastAPI untuk efficient backend development dan integration dengan AI services, yang directly applicable untuk building scalable data-driven applications."
        ]
        
    elif category == "proyek":
        content_responses = [
            "Rush Hour Puzzle Solver definitely yang paling technically challenging dan educational. Project ini require implementation dari multiple pathfinding algorithms - UCS untuk optimal solutions, Greedy Best-First untuk speed, A* untuk balanced approach, dan Dijkstra untuk comprehensive exploration. Biggest challenge adalah optimizing algorithm performance untuk handle complex puzzle configurations without compromising solution quality. Aku juga developed custom heuristic functions dan implement efficient state representation untuk minimize memory usage. Plus, created interactive visualization yang allow users untuk understand algorithm behavior step-by-step, yang require careful balance antara technical accuracy dan user experience.",
            
            "Little Alchemy Solver project yang paling intellectually stimulating karena involve complex graph theory applications. Implementation cover BFS untuk breadth exploration, DFS untuk depth analysis, dan Bidirectional Search untuk optimal pathfinding dalam recipe combination space. Challenge utama adalah handling directed graph dengan dynamic recipe dependencies dan optimizing search strategies untuk minimize computation time. Aku develop sophisticated pruning techniques dan implement memoization untuk avoid redundant computations. Result adalah algorithm yang consistently find optimal recipe paths bahkan untuk complex item combinations yang require dozens of intermediate steps.",
            
            "Setiap algorithmic project memberikan unique insights dan expand understanding tentang computational complexity dan optimization strategies. Rush Hour taught me about state space exploration dan heuristic design, Little Alchemy about graph traversal optimization dan dynamic programming applications, IQ Puzzler Pro about constraint satisfaction dan backtracking efficiency. Collectively, these projects demonstrate ability untuk adapt different algorithmic paradigms to specific problem domains dan develop custom solutions yang account untuk real-world constraints seperti memory limitations dan user experience requirements.",
            
            "Yang bikin proud dari project-project algorithmic ini adalah level of technical sophistication dan attention to both performance dan usability. Bukan cuma implement standard algorithms, tapi juga develop custom optimizations, design efficient data structures, dan create comprehensive testing frameworks untuk validate correctness dan performance. Documentation dan code organization also prioritized untuk ensure maintainability dan knowledge transfer. Each project include detailed analysis tentang time complexity, space complexity, dan comparison dengan alternative approaches untuk demonstrate thorough understanding tentang algorithm design principles."
        ]
        
    # KATEGORI SANTAI - Response yang context-aware berdasarkan specific type
    elif category == "makanan_favorit":
        if specific_type == "why_like":
            content_responses = [
                "Street food Indonesian itu authentic banget dan punya variety yang luas. Martabak manis misalnya, kombinasi rasa manis, tekstur yang fluffy, dan harga yang affordable bikin perfect comfort food. Plus, setiap daerah punya style masing-masing.",
                "Yang bikin aku obsessed sama street food adalah culture dan experience-nya. Bukan cuma soal rasa, tapi social interaction dengan penjual, atmosphere di pinggir jalan, dan feeling nostalgic yang ga bisa didapat di restoran fancy."
            ]
        elif specific_type == "where_to_eat":
            content_responses = [
                "Jakarta punya banyak spot legendary. Sabang sama Pecenongan classic banget buat late night food hunting. Senayan area juga bagus, especially around Gelora Bung Karno. Tiap district punya hidden gems masing-masing.",
                "Prefer spot yang local authentic ketimbang yang touristy. Biasanya hunting di residential areas atau dekat campus. Yang penting crowd-nya mostly local people, itu biasanya indicator kualitas yang good."
            ]
        elif specific_type == "frequency":
            content_responses = [
                "Almost every week sih, especially pas weekend atau abis capek ngerjain projects. Street food itu perfect stress reliever dan ga perlu planning ribet.",
                "Tergantung mood dan budget. Kalau lagi experiment sama koding sampai larut, biasanya jadi excuse buat keluar cari makan enak di sekitar kampus."
            ]
        else:
            content_responses = [
                "Street food Indonesia jadi obsesi. Martabak, sate, ketoprak, semua enak. Jakarta punya spot-spot hidden gems yang seru.",
                "Penggemar berat kuliner jalanan. Gorengan sampai batagor, semuanya punya cerita unik. Sering food hunting sambil ngerjain tugas.",
                "Hobi jelajah street food udah lifestyle. Tiap daerah punya specialty beda. Jakarta, spot favorit around Sabang sama Senayan.",
                "Street food comfort food banget. Pas stress deadline atau stuck coding, keliling cari makan enak jadi refreshing."
            ]
        
    elif category == "lagu_favorit":
        if specific_type == "first_time_hearing":
            content_responses = [
                "Pertama kali kenal musik oldies pas SMA, kakak aku sering putar Air Supply dan Celine Dion. 'Without You' jadi gateway song yang bikin aku appreciate ballad klasik. Voice quality dan emotional depth yang beda banget sama musik mainstream saat itu.",
                "Exposure pertama ke Glenn Fredly waktu ada acara keluarga dan sepupu yang musician recommend. Setelah dengar 'Sekali Ini Saja', langsung jatuh cinta sama style jazz-soul Indonesian. Dari situ mulai explore karya-karya local artist lainnya."
            ]
        elif specific_type == "why_like":
            content_responses = [
                "Oldies punya emotional depth dan musical complexity yang susah dicari di modern music. Lirik-liriknya meaningful, production quality tinggi, dan timeless. Plus, ada nostalgia factor yang bikin relatable di berbagai situasi.",
                "Glenn Fredly special karena pioneer Indonesian jazz-soul. Voice quality luar biasa dan songwriting yang matang. Musik dia sophisticated tapi tetap accessible, perfect balance."
            ]
        elif specific_type == "favorite_lyrics":
            content_responses = [
                "Lirik 'Without You' yang 'I can't live if living is without you' itu classic banget. Simple tapi powerful, dan delivery Air Supply bikin emotional impact-nya kuat.",
                "'Sekali ini saja ku mengalah' from Glenn Fredly itu profound. Ada acceptance dan wisdom dalam simplicity lirik tersebut. Plus melody yang complement perfectly."
            ]
        elif specific_type == "recommendation":
            content_responses = [
                "Kalau suka oldies barat, coba dengar Bee Gees, Chicago, atau Bread. Untuk Indonesian classics, Chrisye dan Iwan Fals legendary. Modern artists yang punya similar vibe: John Mayer atau Adele.",
                "Mutual Spotify actually good idea! Aku curious sama music taste kamu juga. Kalau suka Glenn Fredly, mungkin akan appreciate Tompi atau Andien juga."
            ]
        else:
            content_responses = [
                "Seleranya musik nostalgic & oldies gitu sih. Lagi relate banget sama 'Without You' Air Supply, tapi 'Sekali Ini Saja' Glenn Fredly gak kalah sih. Kalau coding biasanya pakai lo-fi beats atau soundtrack film kayak Star Wars yang bikin suasana lebih intens dan bikin gak ngantuk.",
                "Suka semua lagu, semua generasi. Tapi oldies selalu juaranya, lirik dalem, gak pernah bosenin. Ada emotional connection yang bikin rileks pas overwhelmed.",
                "Mix Indonesia dan barat klasik sih. Glenn Fredly, Noah, Air Supply, Celine Dion, Queen",
                "Musik favorit itu subjective. Tapi kalau ditanya, oldies jadi pilihan prioritas. Bikin relax dan nostalgia. Lo-fi beats juga enak buat coding.",
                "Waduh, banyak banget. Dari oldies kayak Air Supply, Glenn Fredly, Glimpse of Us. Setiap mood ada lagunya."
            ]
        
    elif category == "hobi":
        if specific_type == "when_started":
            content_responses = [
                "Reading habit mulai SMA, pas discover fantasy novels. 'Omniscient Reader Viewpoint' jadi turning point yang bikin aku appreciate storytelling complexity. Traveling baru intensif pas kuliah, chance explore Java dan sekitarnya.",
                "Food hunting udah dari kecil sebenernya, tapi jadi serious hobby pas kuliah. Independence dan exploration nature jadi combine perfectly."
            ]
        elif specific_type == "frequency":
            content_responses = [
                "Reading almost daily, especially pas before sleep atau commute time. Traveling biasanya weekend atau long break. Food hunting spontaneous, tergantung mood dan discover new places.",
                "Balance between intellectual dan physical activities. Kira-kira 60% reading, 30% food exploration, 10% traveling jauh."
            ]
        elif specific_type == "what_interesting":
            content_responses = [
                "Reading bikin aku exposed ke different perspectives dan world-building yang creative. Fantasy genre especially ngajarin aku appreciate complexity dalam system design. Traveling dan food hunting ngasih real-world cultural experience.",
                "Yang menarik adalah how each hobby complement aku punya problem-solving approach. Reading ngasah analytical thinking, traveling ngajarin adaptability, food hunting develop appreciation untuk craftsmanship."
            ]
        else:
            content_responses = [
                "Reading, traveling, food hunting. Novel fantasy kayak Omniscient Reader Viewpoint favorit. Street food exploration nggak bosen.",
                "Balance intellectual dan sensory. Baca novel buat perspective baru, traveling buat cultural exposure. Both influence problem-solving.",
                "Aktivitas yang kasih learning experience. Reading expand imagination, traveling broaden worldview, food hunting appreciate culture.",
                "Hobi somehow connected ke cara kerja. Reading fantasy relate ke system design. Traveling ngajarin adaptability."
            ]
    
    # untuk pertanyaan personal, redirect dengan halus  
    elif category.startswith("personal_"):
        content_responses = [
            "Itu agak personal sih. Mending kita bahas passion aku di data science atau project-project yang lagi dikerjain.",
            "Soal pribadi kurang nyaman share. Kalau tertarik sama technical journey aku, bisa tanya tentang coding experience atau algorithm projects.",
            "Untuk hal personal prefer nggak bahas. Lebih seru discuss proyek data science atau tech stack yang aku pakai."
        ]
    
    else:
        # fallback responses yang simple
        content_responses = [
            "Itu topik yang menarik. Pengalaman dan skill yang aku develop saling connect, dari technical sampai communication skills.",
            "Setiap aspect dari journey aku contribute ke overall capability. Whether technical expertise atau problem-solving approach.",
            "Aku suka sharing tentang experience di tech. Each project dan challenge shaped perspective aku tentang development dan innovation."
        ]
    
    # pilih content dan combine dengan opener
    selected_content = random.choice(content_responses)
    
    # combine dengan spasi yang tepat
    if opener:
        full_response = f"{opener} {selected_content}"
    else:
        full_response = selected_content
    
    # simple closers (jarang dipakai)
    if random.random() > 0.8:  # hanya 20% chance
        simple_closers = [" Gimana?", " Ada lagi?", ""]
        full_response += random.choice(simple_closers)
    
    return full_response

def generate_gibberish_response() -> str:
    responses = [
        "Hmm, maaf aku tidak mengerti pertanyaanmu. Bisa diulangi dengan kata-kata yang lebih jelas?",
        "Sepertinya ada typo dalam pesanmu. Bisa tolong diperjelas? Kamu bisa tanya tentang hobi, musik favorit, atau makanan favoritku.",
        "Aku kurang paham maksudmu. Bisa ditulis ulang dengan lebih jelas? Misalnya tentang hobi, proyek, atau keahlianku.",
        "Wah, aku bingung dengan pertanyaanmu. Mungkin ada kesalahan ketik? Coba tanyakan dengan cara lain tentang diriku.",
        "Maaf, aku tidak mengerti apa yang kamu tulis. Apa kamu ingin bertanya tentang hobi, makanan favorit, atau proyek yang kukerjakan?",
        "Pesanmu sepertinya tidak lengkap atau ada kesalahan ketik. Bisa tolong diperjelas? Aku bisa menjawab tentang diriku, keahlian, atau proyek-proyekku."
    ]
    return random.choice(responses)

def generate_clarification_response() -> str:
    clarifications = [
        "Hmm, pertanyaanmu kurang spesifik nih. Coba tanyakan lebih detail tentang hobi, proyek, keahlian, atau lagu favorit?",
        "Maaf, aku kurang paham maksud pertanyaanmu. Bisa lebih spesifik? Misalnya tentang pendidikan, proyek, atau teknologi yang kugunakan?",
        "Pertanyaanmu terlalu singkat nih. Coba lebih spesifik ya, misalnya tanya tentang lagu favoritku, makanan yang kusuka, atau proyek yang kukerjakan?",
        "Aku kurang yakin apa yang kamu tanyakan. Bisa diperjelas? Kamu bisa tanya tentang pengalamanku, keahlian teknis, atau rencana masa depanku.",
        "Wah, aku tidak yakin apa yang kamu maksud. Mungkin kamu mau tahu tentang mata kuliah favoritku, hobi, atau teknologi yang sering kugunakan?"
    ]
    return random.choice(clarifications)

def create_context_aware_prompt(question: str, context: ConversationContext = None) -> str:
    category = categorize_question(question, context)
    
    base_prompt = f"""
    Kamu adalah asisten pribadi dari {user_profile['nama']} yang cerdas, informatif, dan memiliki kepribadian yang santai. 
    Jawab dengan bahasa Indonesia yang natural dan santai, tapi tetap informatif.
    
    PENTING: Selalu acknowledge pertanyaan user dengan mengutip sebagian pertanyaannya di awal respons, buat conversation terasa interactive dan engaging. Jangan langsung jawab, tapi tunjukkan bahwa kamu understand context dan nuance dari pertanyaan mereka.
    
    Profil dasar:
    - Nama: {user_profile['nama']}
    - Lokasi: {user_profile['lokasi']}
    - Pendidikan: {user_profile['pendidikan']}
    - Pekerjaan saat ini: {user_profile['pekerjaan']}
    - Karakter: {user_profile['karakter']}
    """
    
    # PERBAIKI: hanya tambah konteks jika benar-benar relevan
    if context and context.should_use_context_in_prompt(category):
        previous_question = context.questions_history[-1] if context.questions_history else "Belum ada"
        
        base_prompt += f"""
        Konteks percakapan yang RELEVAN:
        - Kategori terakhir dibahas: {context.last_category}
        - Item yang disebutkan sebelumnya: {', '.join(list(context.mentioned_items)[:3]) if context.mentioned_items else "Belum ada"}
        - Pertanyaan sebelumnya: {previous_question}
        - Tone conversation: {context.conversation_tone}
        
        PENTING: Kamu boleh mereferensikan konteks ini HANYA JIKA pertanyaan saat ini jelas terkait atau berupa follow-up dari percakapan sebelumnya.
        """
    else:
        base_prompt += f"""
        CATATAN: Ini adalah pertanyaan baru yang tidak terkait dengan percakapan sebelumnya. Jawab sebagai topik baru.
        """
    
    # penanganan untuk input gibberish
    if category == "gibberish":
        base_prompt += f"""
        Pertanyaan pengguna terdeteksi sebagai gibberish (teks tidak masuk akal). Berikan respons yang meminta pengguna untuk mengulangi pertanyaan dengan lebih jelas.
        Jangan terlalu teknis dalam menjelaskan bahwa itu gibberish, tapi sarankan secara halus bahwa mungkin ada kesalahan ketik.
        Tawarkan beberapa topik alternatif yang bisa ditanyakan.

        Pertanyaan pengguna: {question}
        
        Jawab dengan bahasa Indonesia yang santai dan natural. Gunakan "aku" untuk merujuk diri sendiri.
        """
        return base_prompt
    
    # jika pertanyaan tidak jelas, minta klarifikasi
    if category == "unclear_question":
        base_prompt += f"""
        Pertanyaan pengguna kurang jelas. Berikan respons yang meminta klarifikasi lebih lanjut.
        Tawarkan beberapa opsi topik yang bisa ditanyakan, seperti hobi, keahlian, proyek, atau lagu favorit.
        
        Pertanyaan pengguna: {question}
        
        Jawab dengan bahasa Indonesia yang santai dan natural. Gunakan "aku" untuk merujuk diri sendiri.
        """
        return base_prompt
    
    # penanganan pertanyaan personal
    if category.startswith("personal_"):
        base_prompt += f"""
        Kamu mendapat pertanyaan yang bersifat personal dan sebaiknya dialihkan. Berikan jawaban dengan format:
        
        1. Acknowledge pertanyaan mereka dengan mengutip sebagian pertanyaannya
        2. Berikan penolakan yang halus untuk menjawab hal personal tersebut
        3. Alihkan pembicaraan ke topik profesional dengan smooth transition
        
        PENTING: Jangan jawab pertanyaan personal apapun, tetapi juga jangan terlalu frontal dalam penolakan. Buat transisi yang natural.
        """
    
    # tambahkan informasi tambahan berdasarkan kategori pertanyaan
    elif category == "rekrutmen":
        base_prompt += f"""
        Pertanyaan tentang rekrutmen atau hiring. Ini pertanyaan yang penting dan perlu dijawab dengan comprehensive dan convincing.
        
        WAJIB:
        1. Acknowledge pertanyaan tentang rekrutmen/hiring di awal
        2. Highlight unique value proposition dengan specific examples
        3. Mention konkrit achievements dan skills
        4. Show passion dan growth mindset
        5. Address concerns tentang experience level jika relevan
        
        Focus pada:
        - Kombinasi theoretical knowledge + practical experience
        - Specific projects dan achievements
        - Learning agility dan adaptability
        - Communication skills dan teamwork
        - Passion for continuous improvement
        
        Proyek unggulan: {user_profile['proyek'][0]}, {user_profile['proyek'][1]}
        Prestasi: {', '.join(user_profile['prestasi'])}
        Pengalaman: {user_profile['pengalaman']}
        """
        
    elif category.endswith("_followup"):
        base_prompt += f"""
        Ini adalah pertanyaan follow-up dari kategori {category.replace('_followup', '')}. 
        
        WAJIB:
        1. Reference kembali ke pertanyaan atau topik sebelumnya
        2. Acknowledge nuance atau challenge dalam pertanyaan
        3. Provide more detailed atau thoughtful analysis
        4. Show self-awareness dan realistic perspective
        
        Context from previous conversation:
        - Last category: {context.last_category if context else 'None'}
        - Previous items discussed: {', '.join(list(context.mentioned_items)[:3]) if context and context.mentioned_items else 'None'}
        """
        
    elif category == "keahlian":
        base_prompt += f"""
        Keahlian: {', '.join(user_profile['keahlian'])}
        Detail keahlian:
        {json.dumps(user_profile['keahlian_detail'], indent=2, ensure_ascii=False)}
        
        Pertanyaan tentang keahlian. WAJIB:
        1. Acknowledge specific aspect yang ditanyakan
        2. Provide concrete examples dan use cases
        3. Explain learning journey atau development process
        4. Show depth of understanding, bukan just surface level
        
        PENTING: Tekankan keahlian di bidang Data Science, bukan AI. Jika menyebutkan AI, sampaikan bahwa itu adalah bagian dari ekosistem Data Science.
        """
        
    elif category == "proyek":
        base_prompt += f"""
        Proyek unggulan:
        1. {user_profile['proyek'][0]}
        2. {user_profile['proyek'][1]}
        3. {user_profile['proyek'][3]}
        
        Detail proyek:
        1. {user_profile['proyek_detail']['Algoritma Pencarian Little Alchemy 2']}
        2. {user_profile['proyek_detail']['Rush Hour Puzzle Solver']}
        3. {user_profile['proyek_detail']['IQ Puzzler Pro Solver']}
        
        Pertanyaan tentang proyek. WAJIB:
        1. Acknowledge interest mereka terhadap proyek
        2. Choose 1-2 most relevant projects untuk discuss in detail
        3. Explain technical challenges dan solutions
        4. Share learnings dan impact from the project
        
        PENTING: Fokuskan pada proyek-proyek algoritma dan puzzle di atas. Explain both technical aspects dan personal growth dari projects tersebut.
        """
        
    # untuk kategori lainnya, maintain same structure...
    elif category == "lagu_favorit":
        base_prompt += f"""
        Lagu favorit:
        {json.dumps(user_profile['lagu_favorit'], indent=2, ensure_ascii=False)}
        
        Pertanyaan tentang musik. WAJIB:
        1. Acknowledge specific aspect tentang musik yang mereka tanyakan
        2. Share personal connection ke lagu-lagu tersebut
        3. Explain mengapa certain songs resonate dengan mu
        4. Maybe mention context kapan usually listen to them
        
        PENTING: Pastikan jawabanmu benar-benar fokus pada lagu dan musik yang disukai, show genuine enthusiasm untuk musik.
        """
        
    elif category == "makanan_favorit":
        base_prompt += f"""
        Makanan favorit:
        {json.dumps(user_profile['makanan_favorit'], indent=2, ensure_ascii=False)}
        
        Pertanyaan tentang makanan/street food. WAJIB:
        1. Acknowledge their interest in your food preferences
        2. Share specific details tentang why you love certain foods
        3. Maybe mention places atau experiences related to the food
        4. Show excitement tentang Indonesian street food culture
        
        PENTING: Focus specifically on street food dan show genuine passion untuk culinary exploration.
        """
    
    else:
        # default handling untuk general questions
        base_prompt += f"""
        Berikan informasi yang relevant dengan pertanyaan. WAJIB:
        1. Acknowledge what they're asking tentang
        2. Connect question to relevant aspects dari profile mu
        3. Provide personal insights atau experiences
        4. Keep it conversational dan engaging
        """
    
    base_prompt += f"""
    Pertanyaan pengguna: {question}
    
    INSTRUCTIONS:
    1. Acknowledge pertanyaan dengan natural, jangan berlebihan
    2. Jawab dengan santai dan to the point
    3. Bahasa Indonesia yang natural, tidak kaku
    4. Kasih contoh spesifik kalau relevan
    5. Gunakan "aku" dan "kamu" 
    6. Max 3-4 kalimat, concise tapi informatif
    7. Jangan terlalu banyak English words
    8. Tone santai, tidak formal atau "lebay"
    
    Contoh yang BENAR: "Oh, soal itu aku punya pengalaman..." 
    Contoh yang SALAH: "That's definitely something worth discussing. From my perspective..."
    """
    
    return base_prompt

def normalize_text(text: str) -> str:
    # hapus spasi berlebih dan standardisasi tanda baca
    cleaned = re.sub(r'\s+', ' ', text)
    cleaned = re.sub(r'\s+\.', '.', cleaned)
    cleaned = re.sub(r'\s+,', ',', cleaned)
    cleaned = re.sub(r',\s+', ', ', cleaned)
    cleaned = re.sub(r'\.\s+', '. ', cleaned)
    cleaned = re.sub(r'\s+!', '!', cleaned)
    cleaned = re.sub(r'!\s+', '! ', cleaned)
    cleaned = re.sub(r'\s+\?', '?', cleaned)
    cleaned = re.sub(r'\?\s+', '? ', cleaned)
    cleaned = re.sub(r'\s+:', ':', cleaned)
    cleaned = re.sub(r':\s+', ': ', cleaned)
    cleaned = re.sub(r'\s+;', ';', cleaned)
    cleaned = re.sub(r';\s+', '; ', cleaned)
    
    return cleaned.strip()

def call_openai_api(prompt):
    logger.info("mengirim permintaan ke openai")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("api key tidak ditemukan")
        raise ValueError("OpenAI API key tidak ditemukan")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Kamu adalah asisten virtual yang membantu menjawab pertanyaan tentang pemilik portfolio dengan cara yang personal, informatif, dan sangat interactive. Selalu acknowledge pertanyaan user dan buat conversation terasa natural."},
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
            logger.error("tidak ada hasil dari openai")
            raise ValueError("Tidak ada hasil dari OpenAI")
        
        raw_response = result["choices"][0]["message"]["content"]
        normalized_response = normalize_text(raw_response)
        return normalized_response
    
    except requests.exceptions.RequestException as e:
        logger.error(f"request error: {str(e)}")
        raise ValueError(f"Error saat berkomunikasi dengan OpenAI: {str(e)}")

@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    try:
        cleanup_old_sessions()
        
        logger.info(f"pertanyaan diterima: {request.question}")
        
        if len(request.question.strip()) < 2:
            return AIResponse(
                response="Pertanyaanmu terlalu singkat. Coba tanyakan lebih detail tentang diriku, seperti hobi, proyek, atau keahlian yang kumiliki.",
                session_id=request.session_id or str(uuid.uuid4())
            )
        
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session ID: {session_id}")
        
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = ConversationContext()
        
        context = conversation_sessions[session_id]
        
        category = categorize_question(request.question, context)
        logger.info(f"Kategori terdeteksi: {category}")
        
        if category == "gibberish":
            response_text = generate_gibberish_response()
            context.update(category, request.question, response_text)
            return AIResponse(response=response_text, session_id=session_id)
        
        # membuat prompt yang lebih kontekstual
        prompt = create_context_aware_prompt(request.question, context)
        
        try:
            response_text = call_openai_api(prompt)
            logger.info("respons diterima dari openai")
            
            context.update(category, request.question, response_text)
            
            return AIResponse(response=response_text, session_id=session_id)
        except Exception as openai_error:
            logger.warning(f"fallback ke mock response: {str(openai_error)}")
            mock_response = await ask_ai_mock(request)
            context.update(category, request.question, mock_response.response)
            mock_response.session_id = session_id
            return mock_response
        
    except Exception as e:
        logger.error(f"error saat memproses permintaan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")

@app.post("/ask-mock", response_model=AIResponse)
async def ask_ai_mock(request: QuestionRequest):
    try:
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = ConversationContext()
        
        context = conversation_sessions[session_id]
        
        question = request.question
        
        if len(question.strip()) < 2:
            return AIResponse(
                response="Pertanyaanmu terlalu singkat. Coba tanyakan lebih detail tentang diriku, seperti hobi, proyek, atau keahlian yang kumiliki.",
                session_id=session_id
            )
        
        if is_gibberish(question):
            response = generate_gibberish_response()
            context.update("gibberish", question, response)
            return AIResponse(response=response, session_id=session_id)
        
        category = categorize_question(question, context)
        question_intent = detect_question_intent(question, context)
        
        # jika pertanyaan tidak jelas, minta klarifikasi
        if category == "unclear_question":
            response = generate_clarification_response()
            context.update(category, question, response)
            return AIResponse(response=response, session_id=session_id)
        
        # generate respons yang lebih interactive
        response_text = generate_interactive_response(question, category, context)
        
        # normalisasi teks respons
        normalized_response = normalize_text(response_text)
        
        context.update(category, question, normalized_response)
        
        return AIResponse(response=normalized_response, session_id=session_id)
        
    except Exception as e:
        logger.error(f"error saat memproses permintaan mock: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")

@app.get("/")
async def root():
    return {"message": "AI Portfolio Backend berjalan. Gunakan endpoint /ask untuk bertanya."}

@app.get("/suggested-followups/{session_id}")
async def get_suggested_followups(session_id: str):
    if session_id not in conversation_sessions:
        raise HTTPException(status_code=404, detail="Session tidak ditemukan")
    
    context = conversation_sessions[session_id]
    suggested_followups = context.get_suggested_followup_questions(3)
    
    return {"suggested_followups": suggested_followups}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)