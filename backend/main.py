from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import logging
import json
import requests
import random
import re

# setup logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# memuat variabel lingkungan
load_dotenv()

# inisialisasi aplikasi
app = FastAPI(title="AI Portfolio Backend")

# konfigurasi cors yang lebih spesifik
frontend_url = os.getenv("FRONTEND_URL", "https://portofolio-danen-frontend.vercel.app")
allowed_origins = [
    frontend_url,
    "http://localhost:3000",  # development
    "https://portofolio-danen-frontend.vercel.app",  # produksi
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model untuk request
class QuestionRequest(BaseModel):
    question: str

# model untuk response
class AIResponse(BaseModel):
    response: str

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
    "hobi": ["Membaca buku sci-fi", "Traveling ke destinasi lokal", "Makan", "Hiking di akhir pekan"],
    "hobi_detail": {
        "Membaca": "Buku favorit termasuk 'Dune' dan karya-karya Ted Chiang",
        "Traveling": "Sudah mengunjungi 8 provinsi di Indonesia dan berencana menambah lagi",
        "Fotografi": "Memiliki akun Instagram khusus untuk hasil foto urban landscape",
        "Hiking": "Mendaki Gunung Rinjani pada 2022 dan Gunung Semeru pada 2023"
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
        "Asisten praktikum Algoritma dan Struktur Data",
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
        "Without You": "Air Supply",
        "Sekali Ini Saja": "Glenn Fredly",
        "Lagu Oldies": "Bee Gees, Westlife, Backstreet Boys"
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

# fungsi untuk mengkategorikan pertanyaan
def categorize_question(question: str) -> str:
    question_lower = question.lower()
    
    # kategori pertanyaan personal yang perlu dialihkan
    if any(word in question_lower for word in ["pacar", "jodoh", "pacaran", "pasangan", "gebetan", "nikah", "menikah", "single", "lajang", "status hubungan"]):
        return "personal_relationship"
    elif any(word in question_lower for word in ["gaji", "salary", "penghasilan", "bayaran", "uang", "kekayaan", "sebulan"]):
        return "personal_financial"
    elif any(word in question_lower for word in ["alamat rumah", "tinggal dimana", "alamat lengkap", "nomor", "kontak", "pribadi"]):
        return "personal_contact"
    elif any(word in question_lower for word in ["umur", "usia", "tanggal lahir", "kapan lahir", "kelahiran"]):
        return "personal_age"
    elif any(word in question_lower for word in ["agama", "kepercayaan", "tuhan", "beribadah"]):
        return "personal_religion"
        
    # kategori umum
    elif any(word in question_lower for word in ["keahlian", "skill", "kemampuan", "ahli", "bisa apa", "bisa apa saja", "jago"]):
        return "keahlian"
    elif any(word in question_lower for word in ["proyek", "project", "karya", "portfolio", "aplikasi", "buat apa", "telah dibuat", "terbaik", "unggulan"]):
        return "proyek"
    elif any(word in question_lower for word in ["tantangan", "challenge", "kesulitan", "masalah", "problem", "hambatan"]):
        return "tantangan_proyek"
    elif any(word in question_lower for word in ["hobi", "suka", "waktu luang", "kegiatan", "aktivitas", "senang"]):
        return "hobi"
    elif any(word in question_lower for word in ["pendidikan", "sekolah", "kuliah", "belajar", "kampus", "universitas", "itb", "masuk itb", "masuk kuliah", "jurusan"]):
        return "pendidikan"
    elif any(word in question_lower for word in ["pelajaran favorit", "mata kuliah favorit", "mata pelajaran"]):
        return "mata_kuliah"
    elif any(word in question_lower for word in ["lokasi", "tinggal", "domisili", "alamat", "kota", "daerah"]):
        return "lokasi"
    elif any(word in question_lower for word in ["prestasi", "pencapaian", "award", "penghargaan", "juara"]):
        return "prestasi"
    elif any(word in question_lower for word in ["lomba", "kompetisi", "contest", "hackathon", "datathon"]):
        return "lomba"
    elif any(word in question_lower for word in ["data", "data science", "analisis data", "big data", "statistik", "machine learning", "ml"]):
        return "data_science"
    elif any(word in question_lower for word in ["ai", "artificial intelligence", "kecerdasan buatan"]):
        return "data_science"  # redirect AI questions to data science
    elif any(word in question_lower for word in ["tool", "alat", "software", "library", "framework", "favorit", "suka pakai"]):
        return "tools"
    elif any(word in question_lower for word in ["karakter", "kepribadian", "sifat", "tipe", "mbti", "orangnya", "pemalu", "extrovert", "introvert"]):
        return "karakter"
    elif any(word in question_lower for word in ["portofolio ini", "website ini", "web ini", "dibuat pakai", "teknologi"]):
        return "portofolio_tech"
    elif any(word in question_lower for word in ["rencana", "masa depan", "target", "tujuan", "cita", "5 tahun"]):
        return "rencana"
    elif any(word in question_lower for word in ["pekerjaan", "kerja", "profesi", "karir", "jabatan"]):
        return "pekerjaan"
    elif any(word in question_lower for word in ["pengalaman", "experience", "lama kerja"]):
        return "pengalaman"
    elif any(word in question_lower for word in ["waktu", "manage", "manajemen", "atur waktu", "produktif"]):
        return "manajemen_waktu"
    elif any(word in question_lower for word in ["stres", "stress", "tekanan", "pressure", "beban", "handle"]):
        return "manajemen_stres"
    elif any(word in question_lower for word in ["cerita", "momen", "pengalaman kuliah", "culture shock", "berkesan"]):
        return "cerita_kuliah"
    elif any(word in question_lower for word in ["organisasi", "berorganisasi", "komunitas", "kepanitiaan"]):
        return "organisasi"
    elif any(word in question_lower for word in ["belajar mandiri", "autodidak", "self-taught", "tutorial"]):
        return "belajar_mandiri"
    elif any(word in question_lower for word in ["kegagalan", "gagal", "failure", "kesalahan", "mistake"]):
        return "belajar_kegagalan"
    elif any(word in question_lower for word in ["tim", "team", "kerja tim", "kolaborasi", "konflik"]):
        return "kerja_tim"
    elif any(word in question_lower for word in ["ngoding", "coding", "kode", "malam", "produktif"]):
        return "kebiasaan_ngoding"
    elif any(word in question_lower for word in ["lagu", "musik", "dengerin", "dengarkan", "playlist"]):
        return "lagu_favorit"
    elif any(word in question_lower for word in ["moto", "motto", "quotes", "quote", "kutipan", "kata-kata"]):
        return "moto_hidup"
    else:
        return "general"

# menyusun prompt yang kontekstual
def create_context_aware_prompt(question: str) -> str:
    category = categorize_question(question)
    
    # base prompt yang selalu ada
    base_prompt = f"""
    Kamu adalah asisten pribadi dari {user_profile['nama']} yang cerdas, informatif, dan memiliki kepribadian yang santai. 
    Jawab dengan bahasa Indonesia yang natural dan santai, tapi tetap informatif.
    
    Profil dasar:
    - Nama: {user_profile['nama']}
    - Lokasi: {user_profile['lokasi']}
    - Pendidikan: {user_profile['pendidikan']}
    - Pekerjaan saat ini: {user_profile['pekerjaan']}
    - Karakter: {user_profile['karakter']}
    """
    
    # penanganan pertanyaan personal
    if category.startswith("personal_"):
        base_prompt += f"""
        Kamu mendapat pertanyaan yang bersifat personal dan sebaiknya dialihkan. Berikan jawaban dengan format:
        
        1. Mulai dengan pernyataan halus bahwa kamu tidak bisa menjawab pertanyaan personal itu (misalnya "Waduh, aku kurang nyaman membahas hal-hal personal seperti itu" atau "Hmm, aku nggak bisa jawab pertanyaan pribadi itu ya")
        2. Lalu alihkan pembicaraan ke topik profesional, seperti keahlian atau proyek (misalnya "Tapi yang jelas, aku bisa cerita kalau...")
        3. Jangan menyebutkan hal-hal personal yang ditanyakan sama sekali dalam jawabanmu
        
        PENTING: Jangan jawab pertanyaan personal apapun, tetapi juga jangan terlalu frontal dalam penolakan.
        """
        
        # tambahkan beberapa topik pengalihan berdasarkan jenis pertanyaan personal
        if category == "personal_relationship":
            base_prompt += f"""
            Alihkan dengan membicarakan fokus pada karir dan proyek. Misalnya: "Yang pasti, saat ini aku lebih fokus mengembangkan karir di bidang data science dan mengerjakan beberapa proyek menarik seperti {user_profile['proyek'][0].split(' - ')[0]} atau {user_profile['proyek'][3].split(' - ')[0]}."
            
            SANGAT PENTING: Jangan mengonfirmasi atau menyangkal status hubungan dalam bentuk apapun.
            """
    
    # tambahkan informasi tambahan berdasarkan kategori pertanyaan
    elif category == "keahlian":
        base_prompt += f"""
        Keahlian: {', '.join(user_profile['keahlian'])}
        Detail keahlian:
        {json.dumps(user_profile['keahlian_detail'], indent=2, ensure_ascii=False)}
        
        Pertanyaan pengguna adalah tentang keahlian. Jawab dengan detail tentang keahlian utama, seberapa mahir, dan bagaimana keahlian tersebut digunakan dalam pekerjaan atau proyek. Berikan contoh konkret.
        
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
        
        Pertanyaan pengguna adalah tentang proyek. Jawab dengan mendeskripsikan salah satu dari proyek algoritma di atas (Algoritma Pencarian Little Alchemy 2, Rush Hour Puzzle Solver, atau IQ Puzzler Pro Solver). Jelaskan tantangan teknis, algoritma yang dipakai, dan hasil yang dicapai.
        
        PENTING: Fokuskan pada proyek-proyek algoritma dan puzzle di atas, bukan proyek lainnya.
        """
    elif category == "tantangan_proyek":
        base_prompt += f"""
        Tantangan proyek:
        {json.dumps(user_profile['tantangan_proyek'], indent=2, ensure_ascii=False)}
        
        Pertanyaan pengguna adalah tentang tantangan dalam proyek. Jelaskan dengan detail tantangan teknis yang dihadapi dalam pengembangan proyek unggulan, terutama proyek algoritma. Ceritakan bagaimana tantangan tersebut diatasi dengan kreativitas dan problem-solving.
        """
    elif category == "hobi":
        base_prompt += f"""
        Hobi: {', '.join(user_profile['hobi'])}
        Detail hobi:
        {json.dumps(user_profile['hobi_detail'], indent=2, ensure_ascii=False)}
        
        Pertanyaan pengguna adalah tentang hobi. Jawab dengan menjelaskan hobi yang disukai, mengapa menyukainya, dan bagaimana meluangkan waktu untuk hobi tersebut. Berikan beberapa cerita menarik terkait hobi.
        """
    elif category == "pendidikan":
        base_prompt += f"""
        Pendidikan: {user_profile['pendidikan']}
        Pendidikan sebelumnya:
        {json.dumps(user_profile['pendidikan_sebelumnya'], indent=2, ensure_ascii=False)}
        
        Pertanyaan pengguna adalah tentang pendidikan. Jawab dengan informasi tentang latar belakang pendidikan, jurusan, mata kuliah favorit, atau pengalaman belajar yang berkesan. Jelaskan juga bagaimana pendidikan mempengaruhi karir.
        """
    elif category == "mata_kuliah":
        base_prompt += f"""
        Mata kuliah favorit: {user_profile['kuliah']['mata_kuliah_favorit']}
        
        Pertanyaan pengguna adalah tentang mata kuliah atau pelajaran favorit. Jelaskan mengapa menyukai mata kuliah tersebut, apa yang menarik, dan bagaimana pengaruhnya terhadap minat di bidang data science dan algoritma.
        """
    elif category == "data_science":
        base_prompt += f"""
        Pengalaman Data Science: 
        - 2 tahun pengalaman di bidang data science
        - Keahlian: analisis data menggunakan pandas, matplotlib, dan scikit-learn
        - Fokus pada pengolahan data, visualisasi, dan pembuatan model prediktif
        - Awal mula belajar: {user_profile['belajar_coding']['data_science']}
        
        Pertanyaan pengguna adalah tentang data science atau AI. Jawab dengan menjelaskan pengalaman dan ketertarikan di bidang data science, bagaimana menggunakan tools seperti Python, pandas, dan scikit-learn dalam proyek. Tekankan bahwa fokus utama adalah data science, bukan AI secara spesifik.
        
        PENTING: Fokuskan pada data science, visualisasi data, dan analisis statistik. Jika pertanyaan tentang AI, jelaskan dalam konteks data science (sebagai bagian dari toolset data science).
        """
    elif category == "tools":
        base_prompt += f"""
        Tools favorit:
        {json.dumps(user_profile['tools_favorit'], indent=2, ensure_ascii=False)}
        
        Pertanyaan pengguna adalah tentang tools yang sering digunakan. Jelaskan tools favorit untuk pengembangan, data science, dan alasan mengapa tools tersebut disukai. Berikan contoh penggunaan tools dalam proyek nyata.
        """
    elif category == "prestasi":
        base_prompt += f"""
        Prestasi: {', '.join(user_profile['prestasi'])}
        
        Pertanyaan pengguna adalah tentang prestasi. Jawab dengan menjelaskan pencapaian penting, penghargaan, atau pengakuan yang pernah diraih. Ceritakan tantangan dan pelajaran yang didapat dari prestasi tersebut.
        """
    elif category == "lomba":
        base_prompt += f"""
        Pengalaman lomba:
        {json.dumps(user_profile['lomba'], indent=2, ensure_ascii=False)}
        
        Pertanyaan pengguna adalah tentang lomba yang pernah diikuti. Ceritakan pengalaman mengikuti lomba, terutama Datavidia UI yang berkesan karena kompleksitasnya. Jelaskan proses, tantangan, dan pembelajaran dari lomba tersebut.
        """
    elif category == "karakter":
        base_prompt += f"""
        Karakter: {user_profile['karakter']}
        Detail kepribadian:
        {json.dumps(user_profile['sifat_detail'], indent=2, ensure_ascii=False)}
        Tipe: {user_profile['personality']['tipe']}
        
        Pertanyaan pengguna adalah tentang kepribadian atau karakter. Jawab dengan menjelaskan sifat-sifat utama, pendekatan dalam bekerja, dan bagaimana karakter tersebut mempengaruhi interaksi profesional dan personal.
        """
    elif category == "portofolio_tech":
        base_prompt += f"""
        Teknologi portofolio:
        {json.dumps(user_profile['portfolio_tech'], indent=2, ensure_ascii=False)}
        
        Pertanyaan pengguna adalah tentang teknologi yang digunakan untuk membangun portofolio ini. Jelaskan stack teknologi yang dipakai (frontend dan backend), alasan pemilihan teknologi tersebut, dan fitur utama dari portofolio.
        """
    elif category == "rencana":
        base_prompt += f"""
        Rencana masa depan: {user_profile['rencana_masa_depan']}
        
        Pertanyaan pengguna adalah tentang rencana atau tujuan masa depan. Jawab dengan menjelaskan visi jangka panjang, rencana karir, proyek impian, atau keahlian yang ingin dikembangkan.
        
        PENTING: Fokuskan pada rencana terkait data science, bukan AI. Jika menyebutkan AI, sampaikan dalam konteks aplikasi data science.
        """
    elif category == "lokasi":
        base_prompt += f"""
        Lokasi: {user_profile['lokasi']}
        
        Pertanyaan pengguna adalah tentang lokasi. Jawab dengan informasi tentang kota tempat tinggal, bagaimana kehidupan di kota tersebut, dan apakah menikmati tinggal di sana.
        """
    elif category == "pekerjaan":
        base_prompt += f"""
        Pekerjaan: {user_profile['pekerjaan']}
        
        Pertanyaan pengguna adalah tentang pekerjaan. Jawab dengan informasi tentang posisi saat ini, tanggung jawab, perusahaan, dan bagaimana perjalanan karir sampai ke posisi sekarang.
        
        PENTING: Tekankan aspek data science dalam pekerjaan, bukan AI.
        """
    elif category == "pengalaman":
        base_prompt += f"""
        Pengalaman: {user_profile['pengalaman']}
        
        Pertanyaan pengguna adalah tentang pengalaman kerja. Jawab dengan informasi tentang lama bekerja di bidang tertentu, proyek yang pernah dikerjakan, dan keterampilan yang didapat dari pengalaman tersebut.
        
        PENTING: Tekankan pengalaman di bidang data science, bukan AI.
        """
    elif category == "manajemen_waktu":
        base_prompt += f"""
        Manajemen waktu: {user_profile['manajemen']['waktu']}
        
        Pertanyaan pengguna adalah tentang manajemen waktu. Jelaskan bagaimana mengelola waktu antara kuliah, proyek, dan kegiatan lain. Berikan tips praktis untuk produktivitas dan efisiensi.
        """
    elif category == "manajemen_stres":
        base_prompt += f"""
        Manajemen stres: {user_profile['manajemen']['stres']}
        
        Pertanyaan pengguna adalah tentang cara mengatasi stres. Jelaskan aktivitas yang dilakukan untuk relaksasi, seperti menonton film horror/romance atau drama Korea. Ceritakan bagaimana pengalaman di ITB melatih ketahanan menghadapi tekanan.
        """
    elif category == "cerita_kuliah":
        base_prompt += f"""
        Pengalaman kuliah: {user_profile['kuliah']['pengalaman_culture_shock']}
        
        Pertanyaan pengguna adalah tentang cerita atau pengalaman di kuliah. Ceritakan tentang culture shock saat masuk ITB, tantangan beradaptasi dengan pace pembelajaran yang cepat, dan bagaimana mengatasi tantangan tersebut.
        """
    elif category == "organisasi":
        base_prompt += f"""
        Pengalaman organisasi: {user_profile['kuliah']['organisasi']}
        
        Pertanyaan pengguna adalah tentang pengalaman organisasi. Ceritakan tentang kepanitiaan Arkavidia di divisi academy yang fokus pada bootcamp data science. Jelaskan peran, tanggung jawab, dan pembelajaran dari pengalaman tersebut.
        """
    elif category == "belajar_mandiri":
        base_prompt += f"""
        Pertanyaan pengguna adalah tentang belajar mandiri. Jelaskan pendekatan dalam belajar secara autodidak, sumber belajar yang digunakan (online courses, tutorial, dokumentasi), dan cara tetap konsisten dalam belajar mandiri.
        """
    elif category == "belajar_kegagalan":
        base_prompt += f"""
        Belajar dari kegagalan: {user_profile.get('manajemen', {}).get('coping_mechanism', 'jangan selalu menuruti coping mechanism diri sendiri')}
        
        Pertanyaan pengguna adalah tentang pelajaran dari kegagalan. Jelaskan pengalaman dari kegagalan akademik, insight yang didapat, dan bagaimana mengatasi coping mechanism yang tidak produktif.
        """
    elif category == "kerja_tim":
        base_prompt += f"""
        Kerja tim: {user_profile['manajemen']['bekerja_tim']}
        
        Pertanyaan pengguna adalah tentang kerja tim atau mengatasi konflik. Jelaskan pendekatan dalam bekerja dengan tim, bagaimana menangani perbedaan pendapat, dan peran yang biasa diambil dalam tim (observer dulu sebelum mengambil inisiatif sebagai leader jika dibutuhkan).
        """
    elif category == "kebiasaan_ngoding":
        base_prompt += f"""
        Kebiasaan ngoding: {user_profile['personality']['kebiasaan_ngoding']}
        
        Pertanyaan pengguna adalah tentang kebiasaan ngoding. Ceritakan preferensi waktu ngoding (terutama malam hari saat pikiran lebih jernih), rutinitas, dan lingkungan yang membuat produktif dalam coding.
        """
    elif category == "lagu_favorit":
        base_prompt += f"""
        Lagu favorit:
        {json.dumps(user_profile['lagu_favorit'], indent=2, ensure_ascii=False)}
        
        Pertanyaan pengguna adalah tentang lagu favorit. Ceritakan lagu yang disukai seperti "Without You" dari Air Supply dan "Sekali Ini Saja" dari Glenn Fredly. Jelaskan juga preferensi untuk lagu oldies dari Bee Gees, Westlife, atau Backstreet Boys saat ngoding.
        """
    elif category == "moto_hidup":
        base_prompt += f"""
        Moto hidup: {user_profile['moto']}
        
        Pertanyaan pengguna adalah tentang moto hidup. Jelaskan moto "Menuju tak terbatas dan melampauinya", makna filosofis di baliknya, dan bagaimana moto tersebut mempengaruhi keputusan dan tindakan sehari-hari.
        """
    else:
        base_prompt += f"""
        Keahlian: {', '.join(user_profile['keahlian'])}
        Hobi: {', '.join(user_profile['hobi'])}
        Proyek unggulan:
        1. {user_profile['proyek'][0]}
        2. {user_profile['proyek'][3]}
        Prestasi: {', '.join(user_profile['prestasi'])}
        Rencana masa depan: {user_profile['rencana_masa_depan']}
        
        Coba tebak apa konteks dari pertanyaan pengguna dan berikan jawaban yang relevan. Hindari jawaban yang terlalu generik. Jika pertanyaan tidak jelas, berikan informasi tentang profil utama dengan singkat dan tawarkan untuk memberikan informasi lebih lanjut tentang topik tertentu.
        
        PENTING: Fokuskan pada data science, bukan AI. Jika membahas keahlian atau proyek, tekankan Algoritma Pencarian Little Alchemy 2.
        """
    
    base_prompt += f"""
    Pertanyaan pengguna: {question}
    
    Jawab dengan bahasa Indonesia yang santai dan alami (tidak kaku), tapi tetap informatif. Gunakan sapaan "aku" saat merujuk diri sendiri dan "kamu" saat merujuk pengguna. Variasikan struktur kalimat untuk terdengar natural. Berikan contoh spesifik dan detail untuk mengilustrasikan poin yang disampaikan. Gunakan sedikit humor yang relevan jika sesuai. Respons max 4-5 kalimat.
    
    PENTING: Pastikan respons kamu tidak mengandung spasi berlebih, pastikan transisi antar kalimat alami dan jelas.
    """
    
    return base_prompt

# fungsi untuk normalisasi teks respons
def normalize_text(text: str) -> str:
    # hapus spasi berlebih dan standardisasi tanda baca
    cleaned = (text
        .replace(r'\s+', ' ')        # ganti multiple spaces dengan single space
        .replace(r'\s+\.', '.')      # hapus spasi sebelum tanda titik
        .replace(r'\s+,', ',')       # hapus spasi sebelum koma
        .replace(r',\s+', ', ')      # standarisasi spasi setelah koma
        .replace(r'\.\s+', '. ')     # standarisasi spasi setelah titik
        .replace(r'\s+!', '!')       # hapus spasi sebelum tanda seru
        .replace(r'!\s+', '! ')      # standarisasi spasi setelah tanda seru
        .replace(r'\s+\?', '?')      # hapus spasi sebelum tanda tanya
        .replace(r'\?\s+', '? ')     # standarisasi spasi setelah tanda tanya
        .replace(r'\s+:', ':')       # hapus spasi sebelum titik dua
        .replace(r':\s+', ': ')      # standarisasi spasi setelah titik dua
        .replace(r'\s+;', ';')       # hapus spasi sebelum titik koma
        .replace(r';\s+', '; ')      # standarisasi spasi setelah titik koma
    )
    
    # gunakan regex untuk hapus spasi berlebih
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s+\.', '.', cleaned)
    cleaned = re.sub(r'\s+,', ',', cleaned)
    cleaned = re.sub(r',\s+', ', ', cleaned)
    cleaned = re.sub(r'\.\s+', '. ', cleaned)
    cleaned = re.sub(r'\s+!', '!', cleaned)
    cleaned = re.sub(r'!\s+', '! ', cleaned)
    cleaned = re.sub(r'\s+\?', '?', cleaned)
    cleaned = re.sub(r'\?\s+', '? ', cleaned)
    
    return cleaned.strip()

# fungsi untuk memanggil OpenAI API
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
            {"role": "system", "content": "Kamu adalah asisten virtual yang membantu menjawab pertanyaan tentang pemilik portfolio dengan cara yang personal, informatif, dan santai."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.7
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
        
        # normalisasi respons sebelum mengembalikan
        raw_response = result["choices"][0]["message"]["content"]
        normalized_response = normalize_text(raw_response)
        return normalized_response
    
    except requests.exceptions.RequestException as e:
        logger.error(f"request error: {str(e)}")
        raise ValueError(f"Error saat berkomunikasi dengan OpenAI: {str(e)}")

# endpoint untuk pertanyaan
@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    try:
        # log pertanyaan
        logger.info(f"pertanyaan diterima: {request.question}")
        
        # membuat prompt yang lebih kontekstual
        prompt = create_context_aware_prompt(request.question)
        
        try:
            # coba panggil openai
            response_text = call_openai_api(prompt)
            logger.info("respons diterima dari openai")
            return AIResponse(response=response_text)
        except Exception as openai_error:
            # jika gagal, gunakan fallback
            logger.warning(f"fallback ke mock response: {str(openai_error)}")
            return await ask_ai_mock(request)
        
    except Exception as e:
        logger.error(f"error saat memproses permintaan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")

# endpoint mock dengan respons yang lebih kontekstual dan format yang lebih baik
@app.post("/ask-mock", response_model=AIResponse)
async def ask_ai_mock(request: QuestionRequest):
    try:
        question = request.question.lower()
        category = categorize_question(question)
        
        # variasi pembuka yang lebih natural
        pembuka = [
            "Hai! ",
            "Oke, ",
            "Hmm, soal itu... ",
            "Menarik pertanyaannya! ",
            "Kalau ditanya soal itu, ",
            "Ah, ",
            "Well, "
        ]
        
        # respons untuk pertanyaan personal
        if category.startswith("personal_"):
            redirects = {
                "personal_relationship": [
                    "Waduh, aku nggak bisa jawab soal kehidupan pribadi kayak gitu hehe. Yang jelas, saat ini aku lagi fokus banget sama karir di data science. Lagi seru ngulik beberapa proyek algoritma seperti Rush Hour Puzzle Solver yang pakai algoritma UCS, Greedy, A*, dan Dijkstra.",
                    "Hmm, aku kurang nyaman bahas hal-hal personal seperti itu. Aku lebih suka cerita tentang proyek Rush Hour Puzzle Solver yang sedang kukerjakan. Ini proyek yang menantang karena perlu implementasi algoritma pathfinding dengan visualisasi interaktif.",
                    "Hehe, aku nggak bisa jawab pertanyaan pribadi begitu. Aku lebih suka fokus ke pengembangan skill di bidang data science. Belakangan ini lagi mendalami pandas dan scikit-learn untuk analisis data."
                ],
                "personal_financial": [
                    "Wah, maaf aku nggak bisa share info finansial seperti itu. Yang bisa aku ceritakan, aku sekarang fokus di data science dan pengembangan algoritma. Proyek terbaru yang kukerjakan adalah Rush Hour Puzzle Solver dengan implementasi berbagai algoritma pathfinding.",
                    "Hmm, soal finansial aku kurang nyaman untuk bahas. Aku lebih senang cerita tentang proyek data science dan pengembangan algoritma seperti Rush Hour Puzzle Solver yang mengimplementasikan UCS, Greedy Best-First Search, A*, dan Dijkstra."
                ],
                "personal_contact": [
                    "Maaf, aku nggak bisa share info kontak personal. Kalau mau tau lebih banyak tentang proyekku, aku lagi fokus di algoritma pencarian untuk game puzzle seperti Little Alchemy 2 Solver yang mengimplementasikan BFS dan DFS.",
                    "Hmm, untuk informasi kontak pribadi aku nggak bisa share ya. Aku senang kalau kamu tertarik dengan kerjaan dan proyekku di bidang data science."
                ],
                "personal_age": [
                    "Hehe, soal umur dan tanggal lahir itu agak personal ya. Yang jelas, aku udah cukup lama berkecimpung di dunia data science dan coding, sekitar 2 tahun pengalaman di pengembangan web dan 1 tahun di data science.",
                    "Daripada bahas umur yang agak personal, mending aku cerita kalau aku punya pengalaman sekitar 2 tahun pengalaman di pengembangan web dan 1 tahun di data science dan lagi fokus mengembangkan skill di data science."
                ],
                "personal_religion": [
                    "Untuk hal-hal pribadi seperti itu, aku kurang nyaman membahasnya. Kalau soal profesional, aku bisa cerita kalau aku fokus di data science dan lagi mengerjakan beberapa proyek menarik tentang algoritma pencarian."
                ]
            }
            
            # pilih respons sesuai kategori personal, atau gunakan default jika kategori tidak spesifik
            if category in redirects:
                responses = redirects[category]
            else:
                responses = [
                    "Hmm, itu pertanyaan yang agak personal, jadi aku nggak bisa jawab dengan spesifik. Yang bisa aku share, aku fokus di bidang data science dan lagi seru mengerjakan beberapa proyek algoritma menarik seperti Rush Hour Puzzle Solver dan Algoritma Pencarian Little Alchemy 2.",
                    "Maaf, untuk hal-hal personal seperti itu aku kurang nyaman membahasnya. Tapi aku senang sharing tentang proyek-proyek data science dan algoritma yang sedang kukerjakan seperti Rush Hour Puzzle Solver yang menggunakan berbagai algoritma pathfinding."
                ]
                
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
        
        # respons berdasarkan kategori yang lebih spesifik
        elif category == "keahlian":
            responses = [
                "Aku paling jago di bidang Python, Data Science, dan Next.js. Terutama untuk data science, aku senang menggunakan pandas dan scikit-learn untuk analisis data. Selain itu, aku juga cukup mahir dengan Python yang kugunakan hampir setiap hari.",
                "Skill utamaku ada di data science dan frontend development. Untuk data science, aku sering pakai Python dengan pandas dan matplotlib untuk visualisasi. Di sisi frontend, Next.js jadi tool favorit untuk bikin aplikasi web interaktif.",
                "Kalau skill teknis, aku cukup percaya diri dengan data science yang sudah kudalami selama 1 tahun. Framework yang sering kupakai adalah pandas, scikit-learn, dan matplotlib untuk visualisasi data."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "proyek":
            responses = [
                "Proyek yang paling kubanggakan adalah Rush Hour Puzzle Solver. Program ini menyelesaikan puzzle Rush Hour menggunakan algoritma pathfinding seperti UCS, Greedy Best-First Search, A*, dan Dijkstra. Dilengkapi dengan CLI dan GUI untuk visualisasi solusi. Proyek ini mengajarkan banyak hal tentang kompleksitas algoritma pencarian dan struktur data yang efisien.",
                "Salah satu proyek favoritku adalah Algoritma Pencarian Little Alchemy 2. Ini proyek implementasi BFS, DFS, dan Bidirectional Search untuk mencari kombinasi recipe dalam permainan. Rasanya puas banget pas algoritma berhasil menemukan kombinasi resep yang optimal.",
                "Aku pernah bikin Rush Hour Puzzle Solver yang cukup menantang. Program ini menyelesaikan puzzle Rush Hour menggunakan algoritma pathfinding seperti UCS, Greedy Best-First Search, A*, dan Dijkstra. Proyek ini jadi salah satu portofolio utama yang sering kutunjukkan ke potential employer karena kompleksitas algoritmanya."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "tantangan_proyek":
            responses = [
                "Tantangan terbesar dalam proyek Rush Hour Puzzle Solver adalah mengoptimalkan algoritma A* dengan heuristik custom agar performa lebih baik. Tadinya algoritma lambat banget untuk puzzle kompleks, tapi setelah optimasi, waktu komputasi berkurang hingga 80%. Tantangan lainnya adalah visualisasi state puzzle yang interaktif dengan library grafis yang terbatas.",
                "Saat mengerjakan Algoritma Pencarian Little Alchemy 2, tantangan utamanya ada di memaksimalkan efisiensi algoritma untuk pencarian kombinasi recipe yang jumlahnya ratusan. Aku harus implementasi Bidirectional search untuk mengatasi bottleneck pada graf hubungan recipe yang super kompleks. Hasilnya, pencarian jadi jauh lebih cepat dibanding BFS standar.",
                "Di proyek IQ Puzzler Pro Solver, tantangan terberatnya adalah state space yang sangat besar karena banyaknya kombinasi yang mungkin. Awalnya, algoritma brute force standard selalu crash karena stack overflow. Akhirnya, berhasil mengatasinya dengan implementasi backtracking dengan pruning yang super efisien."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response

        elif category == "hobi":
            hobby = random.choice(user_profile['hobi'])
            responses = [
                f"Di luar coding, aku suka banget {hobby.lower()}. Menurutku ini bagus untuk refresh otak setelah lama menatap layar dan coding. Kadang juga traveling ke destinasi lokal kalau weekend dan cuacanya bagus.",
                f"Kalau lagi senggang, biasanya aku {hobby.lower()}. Ini jadi semacam 'me time' yang penting untuk balance kerja-hidup. Hobi ini juga sering memberi inspirasi baru untuk proyek-proyek data science.",
                f"Hobi utamaku adalah {hobby.lower()} dan kadang hiking di akhir pekan. Hobi ini benar-benar membantu menyegarkan pikiran dari pekerjaan teknis sehari-hari di data science."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "pendidikan":
            responses = [
                "Aku masih kuliah di ITB jurusan Teknik Informatika, sekarang lagi semester 4 nih. Sebelumnya, aku sekolah di SD Islam Al Azhar 23 Jatikramat, SMP Islam Al Azhar 9 Kemang Pratama, dan SMA Negeri 5 Bekasi. Masuk ITB karena memang tertarik banget sama teknologi dan ilmu komputer.",
                "Saat ini aku masih mahasiswa semester 4 di Teknik Informatika ITB. Dulu aku berjuang keras belajar untuk bisa lolos seleksi SBMPTN. Sebelumnya bersekolah di SD Islam Al Azhar 23, SMP Islam Al Azhar 9, dan SMAN 5 Bekasi. Alhamdulillah keterima di Teknik Informatika yang memang jadi cita-citaku.",
                "Aku masih kuliah semester 4 di ITB jurusan Teknik Informatika. Jenjang pendidikan sebelumnya di SD Islam Al Azhar 23 Jatikramat, dilanjutkan ke SMP Islam Al Azhar 9 Kemang Pratama, lalu SMAN 5 Bekasi. Meskipun tugasnya banyak dan berat, tapi justru di ITB aku belajar banyak tentang problem-solving dan algoritma yang seru."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "mata_kuliah":
            responses = [
                "Pelajaran favoritku sejak dulu adalah Matematika. Di kuliah sekarang juga aku suka banget mata kuliah yang berhubungan dengan matematika dan algoritma. Ada kepuasan tersendiri saat bisa memecahkan problem matematika yang kompleks, dan ilmunya sangat berguna untuk data science yang membutuhkan analisis kuantitatif.",
                "Aku paling suka mata kuliah Matematika, baik waktu sekolah maupun sekarang kuliah. Di ITB, mata kuliah matematika seperti Kalkulus, Aljabar Linear, dan Matematika Diskrit jadi fondasi penting untuk algoritma dan data science. Suka banget momen 'eureka' saat berhasil memecahkan soal matematika yang challenging.",
                "Dari dulu aku memang suka Matematika. Di jurusan Teknik Informatika, kemampuan matematika sangat penting terutama untuk mata kuliah algoritma dan struktur data. Matematika ini juga sangat membantu dalam pemodelan dan analisis data di bidang data science yang sedang kufokuskan."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "data_science":
            responses = [
                "Data science adalah salah satu passion utamaku. Aku mulai belajar dari hal-hal sederhana seperti Excel dan visualisasi data dasar, lalu berkembang ke Python dengan pandas dan scikit-learn. Visualisasi data pakai matplotlib juga jadi bagian yang menyenangkan karena bisa mengubah angka menjadi insight yang mudah dipahami.",
                "Di bidang data science, aku fokus pada pengolahan data dan visualisasi. Awalnya mulai dari Excel dan visualisasi sederhana, sekarang sudah pakai tools yang lebih advanced. Salah satu proyek menarik yang pernah kukerjakan adalah analisis data menggunakan pandas untuk menemukan pola dan tren pada dataset kompleks.",
                "Sebagai data scientist, aku banyak menggunakan Python dengan library seperti pandas, numpy, dan scikit-learn. Perjalananku di data science dimulai dari Excel dan visualisasi data sederhana, yang kemudian berkembang ke analisis yang lebih kompleks. Aku tertarik dengan bagaimana kita bisa mengekstrak informasi berharga dari data mentah."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "tools":
            tools = list(user_profile['tools_favorit'].keys())
            favorite_tool = random.choice(tools)
            tool_desc = user_profile['tools_favorit'][favorite_tool]
            
            responses = [
                f"Tool favoritku untuk coding adalah {favorite_tool} ({tool_desc}). Untuk data science, aku sering pakai Python dengan pandas dan matplotlib. VS Code jadi editor favorit dengan banyak extension yang mempercepat workflow. Jupyter Notebook juga essential untuk eksplorasi data dan eksperimen algoritma.",
                f"Aku paling sering pakai {favorite_tool} untuk development. Selain itu, untuk data science aku selalu pakai pandas dan scikit-learn di Python. Git juga jadi tool wajib untuk version control, terutama saat kolaborasi dengan tim. Figma kadang kupakai untuk wireframing sederhana sebelum coding.",
                f"Kalau soal tools, {favorite_tool} jadi andalanku. Untuk IDE, VS Code dengan berbagai extension-nya bikin produktivitas meningkat. Jupyter Notebook juga sangat membantu untuk eksplorasi data interaktif. Library seperti pandas, matplotlib, dan numpy jadi daily toolkit untuk pekerjaan data science."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "prestasi":
            achievement = random.choice(user_profile['prestasi'])
            
            responses = [
                f"Salah satu pencapaian yang cukup kubanggakan adalah {achievement}. Ini jadi bukti bahwa kerja keras dan passion di bidang data science dan algoritma akan membuahkan hasil. Pengalaman jadi asisten praktikum juga mengajarkan banyak tentang cara menjelaskan konsep teknis dengan lebih mudah dipahami.",
                f"Aku pernah meraih {achievement} yang jadi motivasi untuk terus berkarya. Pencapaian ini mengajarkanku tentang pentingnya kolaborasi dan inovasi dalam teknologi data. Kontribusi ke proyek open source juga membuka jaringan dengan developer lain yang punya minat sama.",
                f"Yang cukup memorable adalah waktu {achievement}. Rasanya jadi validasi atas upaya yang selama ini kulakukan dan membuatku semakin percaya diri dengan arah karir di data science. Kompetisi tersebut menguji kemampuan problem-solving dan implementasi algoritma dalam deadline yang ketat."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "lomba":
            responses = [
                "Aku pernah ikut beberapa lomba, tapi yang paling berkesan adalah Datavidia UI. Lomba ini berkesan karena tingkat kesulitannya yang kompleks dalam analisis data dan machine learning. Kami harus mengolah dataset besar dengan noise, membuat feature engineering kreatif, dan mengoptimalkan model dalam waktu terbatas.",
                "Datavidia UI adalah lomba yang paling berkesan buatku. Lombanya menantang banget karena harus menghasilkan prediksi akurat dari data yang super berantakan. Tim kami harus lembur 2 hari untuk preprocessing data dan fine-tuning model. Meskipun nggak juara, pengalaman dan skillset baru yang kudapat sangat berharga.",
                "Pernah ikut Datavidia UI yang menurutku jadi lomba paling menantang. Challenge-nya soal kompleksitas data yang harus dianalisis dan keterbatasan waktu. Saat itu kami menghadapi dataset dengan banyak missing values dan outliers yang bikin pusing. Tapi justru dari lomba ini aku belajar banyak teknik cleaning dan preprocessing data."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "karakter":
            responses = [
                "Teman-teman biasanya menggambarkan aku sebagai orang yang kreatif, analitis, detail-oriented, dan suka belajar hal baru. Aku juga termasuk orang yang mudah berkenalan dengan orang baru, tidak terlalu pemalu atau introvert. Dalam tim, aku suka membantu mencari solusi dari masalah-masalah teknis.",
                "Aku cenderung mudah bergaul dan berkenalan dengan orang baru, jadi bukan tipe yang pemalu. Karakterku kreatif, analitis, detail-oriented, dan selalu ingin belajar hal baru. Temen-temen bilang aku orangnya problem solver yang suka memecah masalah kompleks jadi bagian-bagian yang lebih mudah ditangani.",
                "Aku bukan orang yang pemalu, malah cenderung extrovert dan mudah berkenalan dengan orang baru. Karakterku adalah kreatif, analitis, detail-oriented, dan selalu penasaran untuk belajar teknologi baru. Dalam diskusi kelompok, aku biasanya aktif menyumbang ide dan mencoba memahami perspektif semua orang."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "portofolio_tech":
            responses = [
                "Portofolio ini dibangun pakai teknologi modern dengan Next.js, TypeScript, dan Tailwind CSS di frontend, serta Python FastAPI di backend. Komponen UI-nya menggunakan Shadcn UI dan ada efek animasi smooth dari Framer Motion. Fitur utamanya adalah asisten AI yang menjawab pertanyaan tentang profilku, ditenagai oleh OpenAI API di backend.",
                "Website portfolio ini dibuat dengan stack Next.js dan TypeScript untuk frontend, dengan styling Tailwind CSS dan komponen Shadcn UI yang rapi. Backend-nya pakai Python FastAPI yang ngehubungin ke OpenAI API. Aku suka kombinasi ini karena Next.js sangat powerful untuk frontend dan Python gampang untuk integrasi AI.",
                "Tech stack buat portofolio ini cukup modern: Next.js + TypeScript + Tailwind CSS untuk frontend, dengan tambahan komponen Shadcn UI yang elegan. Backend-nya pakai Python FastAPI yang terhubung ke OpenAI API. Deploment frontend di Vercel dan backend di Railway. Desainnya mengikuti prinsip mobile-first dengan UI/UX yang clean dan responsif."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "rencana":
            responses = [
                "Ke depannya, setelah lulus aku ingin fokus memperdalam keahlian di bidang data science dan algoritma, lulus dengan prestasi terbaik, dan berkarir di perusahaan teknologi terkemuka. Dalam 5 tahun ke depan, targetku jadi data science specialist yang bisa memimpin proyek-proyek analisis data skala besar.",
                "Rencanaku setelah lulus nanti adalah fokus memperdalam keahlian di bidang data science dan algoritma, lulus dengan prestasi terbaik, dan berkarir di perusahaan teknologi terkemuka. Dalam 5 tahun, aku pengen jadi expert di bidang data engineering dan analytics yang bisa memberikan impact nyata bagi bisnis.",
                "Dalam 5 tahun ke depan, aku berencana untuk menjadi professional di bidang data science dengan spesialisasi di data visualization dan predictive analytics. Fokus utamaku sekarang adalah memperdalam keahlian di bidang data science dan algoritma, lulus dengan prestasi terbaik, dan mendapatkan posisi bagus di perusahaan teknologi terkemuka."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "lokasi":
            responses = [
                "Aku tinggal di Jakarta, Indonesia. Kota ini punya komunitas developer dan data scientist yang aktif dengan banyak meetup dan diskusi menarik. Meskipun kemacetannya kadang bikin stres, tapi Jakarta punya akses bagus ke banyak perusahaan teknologi dan startup.",
                "Saat ini base-ku di Jakarta, Indonesia. Cukup strategis untuk kerja remote maupun onsite dengan berbagai perusahaan teknologi dan data. Banyak acara tech dan data science meetup yang sering kuhadiri buat networking dan update knowledge terbaru di industri.",
                "Domisili di Jakarta, Indonesia. Suka dengan dinamika kota ini meskipun kadang macetnya bikin frustrasi. Tapi dekat dengan banyak tech hub dan komunitas IT yang aktif. Jakarta juga punya banyak coworking space keren yang jadi tempat alternatif saat bosan kerja di rumah."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "pekerjaan":
            responses = [
                "Saat ini aku masih mahasiswa semester 4 di ITB jurusan Teknik Informatika. Belum bekerja full-time, tapi aku aktif sebagai asisten praktikum untuk mata kuliah Berpikir komputasional. Sambil kuliah juga sering bikin proyek-proyek coding untuk portfolio.",
                "Aku masih fokus kuliah di semester 4 Teknik Informatika ITB. Untuk menambah pengalaman, aku jadi asisten praktikum dan kadang ngambil project kecil-kecilan. Masih panjang perjalanannya, tapi aku enjoy banget belajar dan bikin proyek yang menantang.",
                "Belum kerja secara formal karena masih kuliah semester 4 di ITB. Aku aktif di beberapa kegiatan kampus, jadi panitia Arkavidia, dan pernah ikut beberapa lomba terkait hackathon. Fokus utama sekarang masih kuliah sambil mengembangkan skill teknis."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "pengalaman":
            responses = [
                "Meskipun masih kuliah semester 4, aku punya 2 tahun pengalaman di pengembangan web dan 1 tahun di data science. Pengalamanku di data science didapat dari proyek kuliah dan lomba-lomba yang kuikuti. Seru banget bisa belajar langsung dengan praktek.",
                "Pengalaman coding dan data science-ku udah sekitar 2 tahun pengembangan web dan 1 tahun di data science. Meskipun masih kuliah semester 4, aku aktif ikut lomba, bikin proyek, dan jadi asisten praktikum yang nambah banyak jam terbang.",
                "Aku punya pengalaman sekitar 2 tahun pengembangan web dan 1 tahun di data science. Sebagai mahasiswa semester 4, aku dapat banyak pengalaman dari tugas kuliah, lomba-lomba seperti hackathon, dan proyek-proyek kecil yang kukerjakan di luar kuliah."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "manajemen_waktu":
            responses = [
                "Manajemen waktuku di kuliah cukup ketat karena pace di ITB yang super cepat. Aku membagi waktu antara mengerjakan proyek, tugas besar, dan belajar untuk ujian dengan sangat disiplin. Biasanya aku pakai teknik Pomodoro dan time blocking untuk fokus, dan selalu reservasi waktu untuk istirahat dan hobi biar nggak burnout.",
                "Di ITB, manajemen waktu jadi skill krusial buat survive. Aku punya sistem pembagian waktu antara kuliah, tugas, proyek, dan aktivitas lain dengan prioritas yang jelas. Kalender digital dan reminder jadi sahabatku. Kadang aku bikin 'time audit' untuk lihat apakah aktivitasku sesuai dengan prioritas dan goals.",
                "Kunci manajemen waktuku adalah disiplin dan konsistensi. Aku membagi waktu dengan cermat antara mengerjakan proyek, tugas besar, dan persiapan ujian. Hal yang membantu adalah menyiapkan todo list di malam hari untuk esok, dan selalu reservasi 'deep work time' tanpa gangguan untuk tugas yang butuh konsentrasi penuh."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "manajemen_stres":
            responses = [
                "Kuliah di ITB memang udah sering melatih ketahanan menghadapi tekanan, jadi handling stres jadi skill wajib. Cara favoritku mengatasi stres adalah dengan menonton film horror/romance atau drama Korea untuk sejenak escape dari dunia coding. Kadang juga sempatkan olahraga ringan atau jalan-jalan singkat untuk me-refresh pikiran.",
                "Buat handle stres, aku punya jurus jitu: nonton film horror atau romance, atau drama Korea yang seru. Kuliah di ITB dengan tekanannya yang tinggi bikin aku terbiasa dengan deadline dan ekspektasi tinggi. Aku juga percaya pentingnya deep breathing dan short breaks saat coding marathon untuk menjaga kejernihan pikiran.",
                "Dengan tekanan akademik yang tinggi di ITB, aku belajar mengelola stres dengan baik. Biasanya aku menyempatkan menonton film horror/romance atau drama Korea sebagai escape. Kadang juga melakukan hobby lain seperti hiking di akhir pekan. Menurut pengalamanku, penting untuk punya 'mental shutdown time' di antara sesi coding intensif."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "cerita_kuliah":
            responses = [
                "Cerita menarik pas kuliah adalah waktu aku mengalami culture shock karena banyak yang sudah menggeluti dunia IT dari kecil sedangkan aku baru bergabung. Ini bikin aku merasa harus bekerja berkali-kali lipat dari yang lainnya. Tak hanya itu, aku juga kaget ternyata pace pembelajaran materi di ITB sangat amat cepat sehingga harus membagi waktu dengan sangat baik.",
                "Salah satu cerita yang bikin aku kaget pas awal kuliah adalah melihat teman-teman yang sudah jago coding sejak SMP, sementara aku baru mulai serius di SMA. Ini jadi motivation shock yang bikin aku belajar lebih keras. Pace pembelajaran di ITB juga gila-gilaan cepat, dalam seminggu bisa numpuk beberapa tucil (tugas kecil) dan tubes (tugas besar) yang harus dikerjakan paralel.",
                "Pengalaman culture shock terbesar di ITB adalah melihat gap kemampuan yang lebar antar mahasiswa. Banyak yang sudah expert di bidang IT sejak kecil, sementara aku baru mulai. Pace kuliah juga bikin aku kaget, dosen bisa ngejelasin materi super kompleks dalam waktu singkat dan langsung kasih tugas yang bikin melongo. Tapi justru tekanan ini yang bikin aku tumbuh lebih cepat secara teknis."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "organisasi":
            responses = [
                "Aku mengikuti beberapa kepanitiaan, salah satu kepanitiaan yang besar itu Arkavidia dan aku mengisi di divisi academy-nya yang mengelola bootcamp path data science. Pengalaman ini mengajarkan banyak tentang manajemen event, koordinasi tim, dan sharing knowledge tentang data science ke peserta dengan berbagai level pengalaman.",
                "Pengalaman berorganisasi yang berkesan adalah jadi panitia Arkavidia di divisi academy untuk path data science. Tanggung jawabku termasuk menyusun kurikulum bootcamp, koordinasi dengan pemateri, dan memastikan peserta mendapat pengalaman belajar yang optimal. Seru banget bisa sharing knowledge sambil networking dengan profesional di industri.",
                "Salah satu pengalaman berorganisasi yang signifikan adalah terlibat di kepanitiaan Arkavidia, event IT tahunan ITB. Aku di divisi academy yang mengurusi bootcamp data science. Peran ini mengajarkan soft skill berharga seperti leadership, komunikasi, dan project management yang ternyata sangat berguna melengkapi technical skill di dunia IT."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "belajar_mandiri":
            responses = [
                "Untuk belajar mandiri, aku punya strategi mix and match: online courses (Coursera, edX) untuk struktur materi, dokumentasi resmi untuk referensi teknis, dan proyek-proyek kecil untuk praktek. Yang penting adalah konsistensi daily practice, bahkan kalau cuma 20-30 menit sehari. Aku juga suka join forum diskusi dan komunitas untuk dapet insight dari sesama learner.",
                "Belajar mandiri adalah skill vital buat developer. Strategiku adalah kombinasi structured learning via online courses dan exploratory learning dengan bereksperimen pada proyek pribadi. Aku mencatat konsep-konsep penting di Notion yang selalu kureview secara berkala. Selalu set small achievable goals biar ada momentum dan rasa progress.",
                "Kunci belajar mandiri menurut pengalamanku adalah active learning: jangan cuma nonton tutorial, tapi langsung praktek dengan coding. Aku suka bikin proyek kecil untuk mengaplikasikan konsep baru yang kupelajari. Tetap update dengan trends via newsletter dan podcast teknis. Paling penting adalah growth mindset dan sabar dengan diri sendiri saat proses belajar."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "belajar_kegagalan":
            responses = [
                "Pelajaran paling berharga dari kegagalan akademikku adalah jangan selalu menuruti coping mechanism diri sendiri. Dulu aku sering procrastinate dan burnout karena mengerjakan tugas last minute. Sekarang aku lebih aware akan pola self-sabotage dan berusaha membangun habits yang lebih sehat. Setiap kegagalan adalah data points untuk improve strategy belajar.",
                "Kegagalan akademik mengajarkan aku tentang bahaya menuruti coping mechanism yang tidak sehat. Dulu, saat stres dengan deadline, aku sering masuk ke cycle procrastination-panic-rush yang buruk. Sekarang aku belajar menghadapi ketidaknyamanan di awal dan start early pada tugas besar. Kegagalan juga mengajarkan pentingnya seek help dan kolaborasi.",
                "Hal terpenting yang kupelajari dari kegagalan akademik adalah jangan terjebak pada coping mechanism yang destruktif. Aku dulu terjebak dalam pola menunda pekerjaan, lalu kerja marathon yang berujung burnout. Sekarang kupecah tugas besar jadi task-task kecil yang manageable, dan selalu refleksi apa yang worked dan tidak worked dari approach sebelumnya."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "kerja_tim":
            responses = [
                "Dalam kerja tim, aku biasanya liat dulu situasinya: apakah ada yang mau menginisiasi jadi leader, kalau benar-benar gaada baru aku ambil peran itu. Kalau ada konflik, aku cenderung jadi mediator yang fokus ke akar masalah, bukan ke personality. Aku percaya clear communication dan explicit expectations adalah kunci untuk meminimalisir kebanyakan konflik tim.",
                "Gaya kerja tim aku cukup adaptif. Aku lebih suka observe dulu dinamika kelompok, baru ambil peran leader kalau memang dibutuhkan. Untuk konflik, pendekatanku adalah focus on facts, not fault. Aku mencoba mencari common ground dan memastikan semua pihak merasa didengar. Task tracking dan dokumentasi yang rapi juga sangat membantu mengurangi miscommunication.",
                "Aku mengatasi konflik dalam tim dengan pendekatan problem-solving: identifikasi masalah real-nya, cari potential solutions, dan diskusikan trade-offs. Gaya kerjaku adalah observe dulu, baru ambil inisiatif jadi leader kalau memang tidak ada yang mengambil peran tersebut. Aku juga percaya pentingnya clear role dan responsibility distribution dari awal untuk menghindari overlaps dan gaps."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "kebiasaan_ngoding":
            responses = [
                "Terkadang aku memang lebih produktif ngoding malam-malam. Entah kenapa pikiran lebih jernih dan fokus saat dunia lebih sepi. Tapi tetap kuatur supaya tidak mengganggu siklus tidur. Aku biasanya setup IDE dengan dark mode, punya playlist instrumental khusus, dan pastikan punya snack sehat di dekat meja untuk coding marathon.",
                "Iya, aku kadang lebih suka ngoding malam hari karena merasa lebih encer buat mikir dan lebih tenang tanpa distraksi. Tapi nggak selalu sih, biasanya tergantung complexity task-nya. Untuk project yang butuh kreativitas dan problem-solving, malam memang jadi waktu favorit. Setup coding space yang nyaman dan music lofi jadi pendukung penting productive night coding.",
                "Kalau ditanya soal ngoding malam, kadang memang iya. Ada sweet spot dimana otak serasa lebih clear dan creative di jam-jam tertentu di malam hari. Tapi aku juga nggak mau jadi night owl terus karena impacts ke kesehatan. Jadi sekarang lebih ke arah flexible: simple tasks di siang, complex problems di malam, dan tetap prioritasin cukup istirahat."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "lagu_favorit":
            responses = [
                "Untuk saat ini, aku seneng dengerin Without You dari Air Supply, liriknya dalem dan relate banget sama aku. Kalau lagu Indonesia, aku suka denger Glenn Fredly kayak 'Sekali Ini Saja'. Pas lagi ngoding, aku condong ke lagu oldies atau 'old but gold' dengan artis kayak Bee Gees, Westlife, dan Backstreet Boys yang nggak terlalu ganggu fokus.",
                "Lagu favoritku saat ini adalah Without You dari Air Supply, liriknya bener-bener mengena. Juga suka lagu-lagu Glenn Fredly seperti 'Sekali Ini Saja'. Ketika ngoding, playlist-ku biasanya berisi lagu-lagu klasik dari era 90an dan 2000an seperti hits dari Bee Gees, Westlife, atau Backstreet Boys yang bikin mood coding jadi lebih enak.",
                "Aku pecinta musik oldies! Saat ngoding suka dengerin Bee Gees, Westlife, atau Backstreet Boys yang bikin nostalgia. Lagu favorit saat ini Without You dari Air Supply karena liriknya yang dalam dan relate dengan pengalaman pribadi. Untuk lagu Indonesia, aku suka karya-karya Glenn Fredly terutama 'Sekali Ini Saja' yang melodinya bikin nyaman."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        elif category == "moto_hidup":
            responses = [
                "Moto hidupku adalah 'Menuju tak terbatas dan melampauinya'. Bagiku ini adalah filosofi tentang selalu berusaha melampaui batasan yang ada, baik dalam pengembangan teknologi maupun pengembangan diri. Moto ini mengingatkanku untuk tidak cepat puas dengan pencapaian dan selalu mencari cara untuk mengembangkan skill dan knowledge lebih jauh lagi.",
                "Aku punya moto 'Menuju tak terbatas dan melampauinya'. Ini mengingatkanku untuk selalu push boundaries dan jangan terjebak dalam comfort zone. Dalam konteks data science dan programming, moto ini jadi pengingat untuk terus belajar teknologi baru dan mencari solusi yang lebih efisien untuk masalah yang kuhadapi.",
                "'Menuju tak terbatas dan melampauinya' adalah moto yang kupegang. Yap, memang terinspirasi Buzz Lightyear, tapi maknanya dalam bagiku. Ini tentang mindset bahwa selalu ada ruang untuk improvement dan inovasi. Dalam karir tech yang super fast-paced, moto ini jadi reminder untuk stay hungry for knowledge dan berani mengambil challenge baru."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
            
        else:
            # fallback untuk pertanyaan umum
            responses = [
                "Aku seorang mahasiswa dengan fokus di data science. Aku suka mengeksplorasi teknologi baru dan menerapkannya dalam proyek-proyek nyata. Oh iya, salah satu quote favoritku: 'Code is like humor. When you have to explain it, it's bad.'",
                "Secara singkat, aku kreatif, analitis, detail-oriented, dan suka belajar hal baru yang bekerja sebagai mahasiswa. Fokus utamaku ada di data science dan frontend development. Saat ini sedang mengerjakan beberapa proyek algoritma yang menarik.",
                "Aku adalah developer dan data scientist yang berbasis di Jakarta, Indonesia dengan spesialisasi di analisis data dan visualisasi. Selalu berusaha mengembangkan diri dan mencari tantangan baru di dunia teknologi."
            ]
            
            response = random.choice(responses)
            full_response = random.choice(pembuka) + response
        
        # Tambahkan penutup hanya dalam 30% kasus (untuk menghindari kalimat akhir yang terdengar tidak natural)
        if random.random() > 0.7:
            penutup = [
                " Ada lagi yang mau kamu tanyakan?",
                " Gimana menurutmu?",
                " Moga membantu ya!",
                " Ada yang masih kurang jelas?"
            ]
            full_response += random.choice(penutup)
        
        # normalisasi teks respons untuk menghindari spasi berlebih
        normalized_response = normalize_text(full_response)
        
        return AIResponse(response=normalized_response)
    except Exception as e:
        logger.error(f"error saat memproses permintaan mock: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")

# rute health check
@app.get("/")
async def root():
    return {"message": "AI Portfolio Backend berjalan. Gunakan endpoint /ask untuk bertanya."}

# menjalankan aplikasi
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)