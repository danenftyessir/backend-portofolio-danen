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
        self.conversation_tone: str = "neutral"  # track tone dari conversation
    
    def update(self, category: str, question: str, response: str = None):
        self.last_category = category
        self.questions_history.append(question)
        
        if response:
            self.previous_responses.append(response)
            self.extract_mentioned_items(response)
            
            if category not in ["unclear_question", "general"]:
                self.conversation_topics.append(category)
                
            self.generate_potential_followups(category, question, response)
        
        # detect tone dari pertanyaan
        self.detect_conversation_tone(question)
        
        for item in self.mentioned_items:
            if re.search(r'\b' + re.escape(item) + r'\b', question.lower()):
                self.referenced_items[item] += 1
        
        self.last_updated = time.time()
    
    def detect_conversation_tone(self, question: str):
        # deteksi tone conversation untuk adjustment response
        question_lower = question.lower()
        
        # tone categories
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
        # ekstrak item-item penting yang disebutkan dalam respons
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
        "Juara 2 Hackathon Nasional 2022",
        "Asisten praktikum Berpikir Komputasional",
        "Kontributor open source di beberapa proyek React"
    ],
    "lomba": {
        "Datavidia UI": "Pengalaman lomba data science yang paling berkesan karena kompleksitasnya yang menantang",
        "Hackathon Nasional 2022": "Berhasil meraih juara 2 dengan implementasi solusi data-driven untuk masalah transportasi"
    },
    "quotes_favorit": [
        "Code is like humor. When you have to explain it, it's bad.",
        "The best way to predict the future is to create it.",
        "Simplicity is the ultimate sophistication."
    ],
    "moto": "Menuju tak terbatas dan melampauinya",
    "lagu_favorit": {
        "Indonesia": ["Sekali Ini Saja - Glenn Fredly", "Separuh Aku - Noah", "Bukan Rayuan Gombal - Judika"],
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

# fungsi untuk mengecek apakah input adalah gibberish (teks tidak masuk akal)
def is_gibberish(text: str) -> bool:
    if len(text) < 3:
        return False
        
    text = text.lower()
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
    
    for length in range(2, min(5, len(text) // 2 + 1)):
        for start in range(len(text) - length * 2 + 1):
            if text[start:start+length] == text[start+length:start+length*2]:
                return True
    
    words = text.split()
    valid_words = set([
        "hai", "halo", "apa", "siapa", "kenapa", "dimana", "kapan", "bagaimana", "mengapa",
        "kamu", "aku", "saya", "dia", "mereka", "kita", "yang", "dan", "atau", "tapi",
        "untuk", "dari", "dengan", "tanpa", "tentang", "karena", "sebab", "akibat",
        "makanan", "hobi", "lagu", "favorit", "suka", "bisa", "coba", "tolong", 
        "proyek", "coding", "ngoding", "program", "aplikasi", "website", "skill", "keahlian",
        "belajar", "kuliah", "sekolah", "kampus", "itb", "informatika", "komputer",
        "data", "science", "python", "react", "javascript", "java", "next", "nextjs"
    ])
    
    valid_word_count = sum(1 for word in words if word.lower() in valid_words)
    
    if len(words) > 2 and valid_word_count == 0:
        return True
        
    return False

# fungsi untuk mendeteksi intent/sentiment dari pertanyaan
def detect_question_intent(question: str, context: ConversationContext = None) -> str:
    question_lower = question.lower()
    
    # challenging questions (mempertanyakan kemampuan)
    challenging_patterns = [
        r'kenapa (?:saya|aku) harus',
        r'apa (?:yang )?membuat (?:kamu|anda)',
        r'(?:emang|memang|masa) (?:cukup|bisa)',
        r'(?:yakin|serius) (?:bisa|mampu)',
        r'hanya|cuma|saja|doang',
        r'(?:gak|tidak|nggak) (?:cukup|bisa)',
    ]
    
    # reflective questions (minta pendapat/analisis)
    reflective_patterns = [
        r'(?:gimana|bagaimana) (?:menurutmu|pendapatmu)',
        r'kamu (?:rasa|pikir|anggap)',
        r'menurut (?:kamu|anda)',
        r'apa (?:pendapat|opini)',
    ]
    
    # comparison questions (membandingkan)
    comparison_patterns = [
        r'(?:dibanding|dibandingkan)',
        r'lebih (?:baik|bagus)',
        r'versus|vs',
        r'mana yang',
    ]
    
    # follow up questions
    followup_patterns = [
        r'(?:terus|lalu|kemudian)',
        r'(?:selain|kecuali) (?:itu|dari)',
        r'balik ke',
        r'kembali ke',
        r'oh (?:gitu|begitu)',
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
    
    for pattern in followup_patterns:
        if re.search(pattern, question_lower):
            return "followup"
    
    return "neutral"

# categories keywords untuk deteksi yang lebih nuanced
category_keywords = {
    "lagu_favorit": [
        "lagu", "musik", "dengerin", "dengarkan", "nyanyi", "penyanyi", "band", "playlist",
        "genre", "album", "konser", "artist", "artis", "musisi", "spotify", "instrumental", 
        "mendengarkan", "lagu favorit", "lagu kesukaan", "musik favorit", "enak didengerin",
        "air supply", "glenn fredly", "without you", "sekali ini saja", "celine dion", 
        "oldies", "lo-fi", "soundtrack", "pop", "ballad"
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
    if context:
        logger.info(f"Context - Last category: {context.last_category}")
        logger.info(f"Context - Mentioned items: {context.mentioned_items}")
    
    # deteksi gibberish (teks tidak masuk akal)
    if is_gibberish(question):
        logger.info(f"Detected gibberish: {question}")
        return "gibberish"
    
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
    
    # deteksi follow-up dengan referensi ke percakapan sebelumnya
    if context and context.questions_history:
        followup_patterns = [
            r'balik ke (?:sebelumnya|tadi)',
            r'kembali ke (?:pertanyaan|topik) (?:sebelumnya|tadi)',
            r'(?:lanjutan|kelanjutan) dari',
            r'(?:terus|lalu) (?:gimana|bagaimana)',
            r'(?:emang|memang) (?:cukup|bisa)',
            r'(?:yakin|serius) (?:gak|tidak)',
            r'hanya (?:dengan|pakai)',
            r'cuma (?:dengan|pakai)',
        ]
        
        for pattern in followup_patterns:
            if re.search(pattern, question):
                # gunakan kategori sebelumnya atau deteksi berdasarkan konten
                if context.last_category:
                    return context.last_category + "_followup"
    
    # pengecekan kategori personal dulu
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
        
        # berikan bobot tambahan jika kategori sama dengan previous category
        if context and context.last_category == category:
            matches += 1
        
        # cek apakah kategori ini berkaitan dengan item yang disebutkan sebelumnya
        if context and context.mentioned_items:
            for item in context.mentioned_items:
                if any(re.search(r'\b' + re.escape(item) + r'\b', keyword) for keyword in keywords):
                    matches += 0.5
        
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
    
    # cek apakah pertanyaan terlalu pendek atau kurang jelas
    if len(question.split()) < 3:
        return "unclear_question"
    
    return "general"

def generate_interactive_response(question: str, category: str, context: ConversationContext = None) -> str:
    # deteksi intent untuk konteks, tapi jangan ekspos ke user
    question_intent = detect_question_intent(question, context)
    
    # opener yang natural tanpa format array
    simple_openers = [
        "Oh iya,",
        "Soal itu,",
        "Hm,",
        "Kalau itu,",
        "",  # kadang langsung jawab tanpa opener
    ]
    
    # untuk follow-up questions
    if question_intent == "followup" and context and context.last_category:
        followup_openers = [
            "Oh, balik ke topik tadi ya?",
            "Iya, soal itu tadi...",
            "Right,",
            ""
        ]
        opener = random.choice(followup_openers)
    else:
        opener = random.choice(simple_openers)
    
    # generate content yang natural berdasarkan kategori
    # KATEGORI SERIUS - Response panjang dan detail
    if category == "rekrutmen":
        content_responses = [
            # Variasi 1: Focus pada kombinasi teori-praktik
            "Aku bawa kombinasi teori dan praktek di data science yang cukup solid. Udah terapin langsung di proyek kayak Rush Hour Puzzle Solver dengan implementasi algoritma pathfinding kompleks seperti UCS, A*, dan Dijkstra. Plus pengalaman dari berbagai lomba termasuk Datavidia UI yang ngasah kemampuan handle dataset kompleks, dan jadi asisten praktikum yang ngelatih communication skills. Yang bikin aku beda adalah dedikasi buat terus belajar teknologi baru dan nggak takut hadapi tantangan yang belum pernah dicoba sebelumnya.",
            
            # Variasi 2: Focus pada mindset problem-solving
            "Yang bikin aku menonjol adalah cara pendekatan masalah yang sistematis dan analitis. Dengan pengalaman 2 tahun web development dan 1 tahun fokus data science, aku udah develop mindset untuk breakdown complex problems jadi manageable pieces. Contohnya di proyek Little Alchemy Solver, aku harus implement BFS, DFS, dan Bidirectional Search untuk optimize recipe search di graph yang kompleks. Bukan cuma skill teknis yang solid, tapi juga kemampuan komunikasikan insight dan findings ke stakeholder dengan cara yang mudah dipahami. Plus track record akademik yang konsisten dan keterlibatan aktif di organisasi tech kayak Arkavidia.",
            
            # Variasi 3: Focus pada track record
            "Track record aku di berbagai domain cukup membuktikan versatility dan konsistensi. Mulai dari academic projects yang challenging, participation di kompetisi data science, sampai kontribusi open source di proyek React. Yang paling menonjol adalah implementasi algoritma advanced di berbagai puzzle solver - dari state space optimization sampai heuristic design. Experience ini ngasih aku solid foundation dalam algorithm design dan performance tuning. Plus, background sebagai asisten praktikum dan active member di komunitas tech bikin aku comfortable dalam collaborative environment dan knowledge sharing.",
            
            # Variasi 4: Focus pada passion dan learning agility
            "Passion aku di data science itu genuine dan bukan cuma trend following. Ini terbukti dari consistency dalam deliver hasil berkualitas, baik di tugas kuliah yang demanding, participation di lomba-lomba competitive, maupun contribution ke proyek open source. Yang jadi kekuatan utama adalah learning agility - aku cepat adapt dengan teknologi baru dan selalu penasaran sama challenge yang belum pernah encountered sebelumnya. Contohnya, dalam waktu singkat aku bisa master different algorithm paradigms dan apply mereka ke real-world problems dengan effective optimization strategies.",
            
            # Variasi 5: Focus pada value yang dibawa
            "Aku nggak cuma bawa hard skills yang solid, tapi juga fresh perspective dari dunia akademik dan exposure ke latest research trends. Kombinasi theoretical knowledge di algorithm design dengan hands-on experience di practical implementation bikin aku bisa bridge gap antara research dan application. Skill set aku cover dari data preprocessing dengan pandas, modeling dengan scikit-learn, sampai full-stack development dengan Next.js. Yang penting, aku selalu fokus deliver actionable insights dan maintainable solutions, bukan cuma proof of concept yang impressive tapi susah di-scale.",
            
            # Variasi 6: Focus pada collaborative approach
            "Aku tipe yang thrive dalam collaborative environment dan genuine enjoy sharing knowledge dengan team members. Experience sebagai asisten praktikum dan involvement di organizing committees ngajarin aku gimana cara effective communication dengan diverse backgrounds. Technical skills aku solid - proven di berbagai algorithmic projects dan consistent academic performance - tapi yang lebih valuable adalah ability untuk facilitate knowledge transfer dan mentor junior team members. Plus, aku always open untuk feedback dan continuous improvement, yang crucial dalam fast-paced tech environment.",
            
            # Variasi 7: Focus pada unique combination
            "Kombinasi unik aku adalah strong academic foundation yang dibalance dengan practical project experience dan soft skills development. Aku nggak cuma understand algorithms secara theoretical, tapi udah implement dan optimize mereka untuk real-world constraints. Portfolio projects kayak IQ Puzzler Pro Solver dan Rush Hour Puzzle menunjukkan kemampuan handle complex state spaces dan develop efficient solutions. Plus, active involvement di tech community dan teaching experience bikin aku comfortable dengan knowledge dissemination dan cross-functional collaboration.",
            
            # Variasi 8: Focus pada problem-solving capability  
            "Kelebihan utama aku adalah systematic approach dalam problem decomposition dan solution architecture. Terbukti dari success rate di berbagai challenging projects yang require both algorithmic thinking dan engineering practicality. Aku bisa efficiently analyze requirements, design optimal approaches, dan implement robust solutions yang scalable. Experience covering different domains - dari graph algorithms sampai web development - kasih aku versatile skill set yang applicable ke various business problems. Plus mindset yang always question assumptions dan explore alternative approaches untuk ensure optimal outcomes."
        ]
    
    elif category in ["keahlian_followup", "pengalaman_followup"]:
        if question_intent == "challenging":
            content_responses = [
                # Response detail untuk pertanyaan challenging tentang pengalaman
                "'Cukup' itu memang relatif, tapi aku confident dengan depth dari pengalaman yang udah dijalani. Dalam 2-3 tahun terakhir, aku nggak cuma belajar surface level tapi bener-bener deep dive ke setiap technology stack dan methodology. Contohnya di Rush Hour Solver, aku spend significant time untuk optimize algorithm performance dari naive approach sampai sophisticated heuristic design. Setiap project ngajarin unique challenges - dari memory management di large state spaces sampai UI/UX considerations untuk complex visualizations. Plus, academic environment di ITB constantly expose aku ke cutting-edge research dan best practices yang immediately applicable ke practical problems.",
                
                "Fair point untuk questioning that. Learning curve di tech industry memang steep dan landscape-nya terus berubah rapidly. Tapi yang aku notice adalah foundation yang solid di fundamentals bikin adaptation ke new technologies jadi much smoother. Pengalaman sekarang udah cover diverse areas - algorithm design, data analysis, web development, dan system optimization. Yang penting adalah aku develop strong problem-solving methodology yang transferable across domains. Plus, habit untuk continuous learning dan engagement dengan tech community ensure aku stay updated dengan industry trends dan emerging technologies.",
                
                "Honestly, aku acknowledge bahwa experience aku belum se-extensive senior developers dengan decades of industry background. Tapi what I lack in years, aku compensate dengan intensity dan breadth of learning. Every single project aku treat sebagai comprehensive learning opportunity - from initial research phase sampai post-implementation analysis. Academic setting di ITB juga provide unique advantages seperti access ke latest research, structured approach to problem-solving, dan opportunity untuk experiment dengan novel approaches without commercial pressure constraints."
            ]
        else:
            content_responses = [
                # Response detail untuk follow-up tentang keahlian
                "Pengalaman aku di data science specifically focus pada end-to-end implementation yang solve real problems. Mulai dari data acquisition dan preprocessing, exploratory analysis untuk understand patterns, feature engineering untuk optimize model performance, sampai deployment considerations untuk production environment. Portfolio projects demonstrate kemampuan handle different types of challenges - optimization problems di puzzle solvers, predictive modeling di finance tracker, dan algorithm visualization untuk educational purposes.",
                
                "Skill development journey aku quite structured dan progressive. Dimulai dari solid foundation di programming fundamentals, then expanding ke specialized areas seperti machine learning, algorithm optimization, dan web development. Each phase build upon previous knowledge - web dev experience help dengan deployment considerations, algorithm background crucial untuk model optimization, dan data analysis skills essential untuk business insight generation. Current focus adalah integrating AI capabilities dengan practical applications, seperti yang demonstrated di portfolio ini."
            ]
    
    elif category == "keahlian":
        content_responses = [
            # Response detail untuk keahlian profesional
            "Skill utama aku ada di intersection antara data science dan web development, dengan strong foundation di Python dan JavaScript ecosystems. Untuk data science, aku proficient dengan pandas untuk data manipulation, scikit-learn untuk machine learning models, matplotlib dan seaborn untuk visualization, dan numpy untuk numerical computing. Web development side, aku experienced dengan Next.js dan React untuk frontend, plus kemampuan integrate dengan backend services. Yang crucial adalah understanding tentang when to use which tool - efficiency dan hasil akhir selalu jadi primary consideration dalam technology selection.",
            
            # Variasi 2: Intersection focus dengan detail
            "Specialization aku adalah combining data science capabilities dengan modern web development, yang relatively unique combination. Python jadi daily driver untuk all data-related tasks - from preprocessing messy datasets sampai building predictive models. Next.js dan React ecosystem aku gunakan untuk create interactive applications yang showcase data insights effectively. Portfolio ini perfect example dari integration tersebut - AI backend dengan FastAPI, modern frontend dengan TypeScript, dan seamless user experience. Plus, aku comfortable dengan deployment strategies untuk both data science models dan web applications.",
            
            # Variasi 3: Practical experience dengan detail
            "Yang bikin aku confident adalah extensive hands-on experience dengan tools yang aku claim sebagai expertise. Pandas bukan cuma untuk basic data manipulation, tapi advanced operations seperti multi-index handling, time series analysis, dan performance optimization untuk large datasets. Scikit-learn usage cover dari classical algorithms sampai ensemble methods, dengan understanding tentang hyperparameter tuning dan cross-validation strategies. React development includes state management dengan complex component architectures, performance optimization dengan memoization, dan integration dengan various APIs dan databases.",
            
            # Variasi 4: Learning approach dengan detail
            "Approach aku dalam skill development adalah depth over breadth, dengan focus ke technologies yang proven valuable dan versatile. Python ecosystem dipilih karena extensive libraries untuk data science, machine learning, dan general programming. JavaScript dengan React/Next.js chosen karena powerful untuk building modern, interactive applications. But aku always open untuk explore new technologies kalau memang solve specific problems better. Recent exploration includes FastAPI untuk efficient backend development dan integration dengan AI services, yang directly applicable untuk building scalable data-driven applications."
        ]
        
    elif category == "proyek":
        content_responses = [
            # Response detail untuk proyek profesional
            "Rush Hour Puzzle Solver definitely yang paling technically challenging dan educational. Project ini require implementation dari multiple pathfinding algorithms - UCS untuk optimal solutions, Greedy Best-First untuk speed, A* untuk balanced approach, dan Dijkstra untuk comprehensive exploration. Biggest challenge adalah optimizing algorithm performance untuk handle complex puzzle configurations without compromising solution quality. Aku juga developed custom heuristic functions dan implement efficient state representation untuk minimize memory usage. Plus, created interactive visualization yang allow users untuk understand algorithm behavior step-by-step, yang require careful balance antara technical accuracy dan user experience.",
            
            # Variasi 2: Little Alchemy focus dengan detail
            "Little Alchemy Solver project yang paling intellectually stimulating karena involve complex graph theory applications. Implementation cover BFS untuk breadth exploration, DFS untuk depth analysis, dan Bidirectional Search untuk optimal pathfinding dalam recipe combination space. Challenge utama adalah handling directed graph dengan dynamic recipe dependencies dan optimizing search strategies untuk minimize computation time. Aku develop sophisticated pruning techniques dan implement memoization untuk avoid redundant computations. Result adalah algorithm yang consistently find optimal recipe paths bahkan untuk complex item combinations yang require dozens of intermediate steps.",
            
            # Variasi 3: Learning impact dengan detail  
            "Setiap algorithmic project memberikan unique insights dan expand understanding tentang computational complexity dan optimization strategies. Rush Hour taught me about state space exploration dan heuristic design, Little Alchemy about graph traversal optimization dan dynamic programming applications, IQ Puzzler Pro about constraint satisfaction dan backtracking efficiency. Collectively, these projects demonstrate ability untuk adapt different algorithmic paradigms to specific problem domains dan develop custom solutions yang account untuk real-world constraints seperti memory limitations dan user experience requirements.",
            
            # Variasi 4: Technical depth dengan detail
            "Yang bikin proud dari project-project algorithmic ini adalah level of technical sophistication dan attention to both performance dan usability. Bukan cuma implement standard algorithms, tapi juga develop custom optimizations, design efficient data structures, dan create comprehensive testing frameworks untuk validate correctness dan performance. Documentation dan code organization also prioritized untuk ensure maintainability dan knowledge transfer. Each project include detailed analysis tentang time complexity, space complexity, dan comparison dengan alternative approaches untuk demonstrate thorough understanding tentang algorithm design principles."
        ]
        
    # KATEGORI SANTAI - Response singkat dan casual
    elif category == "makanan_favorit":
        content_responses = [
            "Street food Indonesia jadi obsesi. Martabak, sate, ketoprak, semua enak. Jakarta punya spot-spot hidden gems yang seru.",
            "Penggemar berat kuliner jalanan. Gorengan sampai batagor, semuanya punya cerita unik. Sering food hunting sambil ngerjain tugas.",
            "Hobi jelajah street food udah lifestyle. Tiap daerah punya specialty beda. Jakarta, spot favorit around Sabang sama Senayan.",
            "Street food comfort food banget. Pas stress deadline atau stuck coding, keliling cari makan enak jadi refreshing."
        ]
        
    elif category == "lagu_favorit":
        content_responses = [
            "Selera musik nostalgic. 'Without You' Air Supply, 'Sekali Ini Saja' Glenn Fredly. Coding pakai lo-fi beats atau soundtrack film.",
            "Suka ballad 90an. Lirik dalam, melodi timeless. Ada emotional connection yang bikin rileks pas overwhelmed.",
            "Mix Indonesia dan barat klasik. Glenn Fredly, Noah dari lokal. Air Supply, Celine Dion internasional. Mostly pop ballad.",
            "Musik punya fungsi beda. Ballad buat relaxing, instrumental buat coding, oldies buat mood booster. Playlist organized by activity."
        ]
        
    elif category == "hobi":
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
    
    # tambah informasi tentang percakapan sebelumnya jika tersedia
    if context and (context.questions_history or context.mentioned_items):
        previous_question = context.questions_history[-1] if context.questions_history else "Belum ada"
        
        base_prompt += f"""
        Konteks percakapan:
        - Kategori terakhir dibahas: {context.last_category if context.last_category else "Belum ada"}
        - Item yang disebutkan sebelumnya: {', '.join(context.mentioned_items) if context.mentioned_items else "Belum ada"}
        - Pertanyaan sebelumnya: {previous_question}
        - Tone conversation: {context.conversation_tone}
        
        PENTING: Jika pertanyaan saat ini sepertinya mereferensikan atau follow-up dari percakapan sebelumnya, WAJIB acknowledge connection tersebut dan respond dalam konteks yang relevan.
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
        - Previous items discussed: {', '.join(context.mentioned_items) if context and context.mentioned_items else 'None'}
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