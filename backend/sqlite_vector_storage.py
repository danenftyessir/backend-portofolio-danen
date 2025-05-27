import sqlite3
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import Counter
import math
import re

logger = logging.getLogger(__name__)

class SQLiteVectorStorage:
    """
    lightweight vector storage menggunakan sqlite
    menggunakan tf-idf tanpa library eksternal untuk minimalkan memori
    """
    
    def __init__(self, db_path: str = "portfolio_vectors.db"):
        self.db_path = db_path
        self.conn = None
        self.vocabulary = {}
        self.idf_scores = {}
        self.doc_count = 0
        self._init_database()
    
    def _init_database(self):
        """inisialisasi database sqlite"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            
            # tabel dokumen
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    category TEXT,
                    title TEXT,
                    content TEXT,
                    keywords TEXT,
                    vector TEXT
                )
            """)
            
            # tabel vocabulary untuk tf-idf
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vocabulary (
                    word TEXT PRIMARY KEY,
                    doc_frequency INTEGER,
                    idf_score REAL
                )
            """)
            
            # index untuk performa
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON documents(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_word ON vocabulary(word)")
            
            self.conn.commit()
            logger.info("✅ sqlite database initialized")
            
        except Exception as e:
            logger.error(f"❌ error initializing sqlite: {e}")
            raise
    
    def _tokenize(self, text: str) -> List[str]:
        """tokenisasi sederhana tanpa nltk"""
        # lowercase dan extract words
        text = text.lower()
        # hapus karakter khusus, keep alphanumeric
        words = re.findall(r'\b[a-z0-9]+\b', text)
        # filter stopwords sederhana
        stopwords = {'dan', 'atau', 'yang', 'di', 'ke', 'dari', 'untuk', 'dengan', 'adalah', 'ini', 'itu'}
        return [w for w in words if len(w) > 2 and w not in stopwords]
    
    def _calculate_tf(self, words: List[str]) -> Dict[str, float]:
        """hitung term frequency"""
        word_count = Counter(words)
        total_words = len(words)
        
        if total_words == 0:
            return {}
        
        # normalized tf
        return {word: count / total_words for word, count in word_count.items()}
    
    def _update_vocabulary(self):
        """update idf scores setelah semua dokumen di-load"""
        cursor = self.conn.cursor()
        
        # hitung total dokumen
        cursor.execute("SELECT COUNT(*) FROM documents")
        self.doc_count = cursor.fetchone()[0]
        
        if self.doc_count == 0:
            return
        
        # update idf untuk setiap word
        cursor.execute("SELECT word, doc_frequency FROM vocabulary")
        vocab_data = cursor.fetchall()
        
        for word, doc_freq in vocab_data:
            # idf = log(total_docs / doc_frequency)
            idf = math.log(self.doc_count / doc_freq) if doc_freq > 0 else 0
            cursor.execute(
                "UPDATE vocabulary SET idf_score = ? WHERE word = ?",
                (idf, word)
            )
        
        self.conn.commit()
        
        # load ke memory untuk akses cepat
        cursor.execute("SELECT word, idf_score FROM vocabulary")
        self.idf_scores = dict(cursor.fetchall())
    
    def _vector_to_string(self, vector: Dict[str, float]) -> str:
        """konversi vector dict ke string untuk storage"""
        # simpan hanya top-k terms untuk hemat space
        top_k = 50
        sorted_terms = sorted(vector.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return json.dumps(dict(sorted_terms))
    
    def _string_to_vector(self, vector_str: str) -> Dict[str, float]:
        """konversi string ke vector dict"""
        return json.loads(vector_str) if vector_str else {}
    
    def add_documents(self, documents: List[Dict]) -> bool:
        """tambah dokumen ke storage"""
        try:
            cursor = self.conn.cursor()
            
            # track word document frequency
            word_doc_freq = Counter()
            
            for doc in documents:
                # extract fields
                doc_id = doc.get('id', '')
                category = doc.get('category', 'general')
                title = doc.get('title', '')
                content = doc.get('content', '')
                keywords = doc.get('keywords', [])
                
                # gabung semua text
                full_text = f"{title} {content} {' '.join(keywords)}"
                
                # tokenize
                words = self._tokenize(full_text)
                unique_words = set(words)
                
                # update word doc frequency
                for word in unique_words:
                    word_doc_freq[word] += 1
                
                # hitung tf vector
                tf_vector = self._calculate_tf(words)
                
                # boost keywords
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in tf_vector:
                        tf_vector[keyword_lower] *= 2.0  # keyword boost
                
                # simpan dokumen
                cursor.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, category, title, content, keywords, vector)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    category,
                    title,
                    content,
                    json.dumps(keywords),
                    self._vector_to_string(tf_vector)
                ))
            
            # update vocabulary
            for word, doc_freq in word_doc_freq.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO vocabulary (word, doc_frequency)
                    VALUES (?, ?)
                """, (word, doc_freq))
            
            self.conn.commit()
            
            # update idf scores
            self._update_vocabulary()
            
            logger.info(f"✅ added {len(documents)} documents to sqlite storage")
            return True
            
        except Exception as e:
            logger.error(f"❌ error adding documents: {e}")
            return False
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """hitung cosine similarity antara 2 vector"""
        # cari common words
        common_words = set(vec1.keys()) & set(vec2.keys())
        
        if not common_words:
            return 0.0
        
        # dot product
        dot_product = sum(vec1[w] * vec2[w] for w in common_words)
        
        # magnitude
        mag1 = math.sqrt(sum(v * v for v in vec1.values()))
        mag2 = math.sqrt(sum(v * v for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def search(self, query: str, top_k: int = 3, category_filter: Optional[str] = None) -> List[Dict]:
        """search dokumen berdasarkan query"""
        try:
            # tokenize query
            query_words = self._tokenize(query)
            if not query_words:
                return []
            
            # hitung query tf-idf vector
            query_tf = self._calculate_tf(query_words)
            query_vector = {}
            
            for word, tf in query_tf.items():
                if word in self.idf_scores:
                    query_vector[word] = tf * self.idf_scores[word]
            
            if not query_vector:
                return []
            
            # query dokumen
            cursor = self.conn.cursor()
            
            if category_filter:
                cursor.execute("""
                    SELECT id, category, title, content, keywords, vector
                    FROM documents
                    WHERE category = ?
                """, (category_filter,))
            else:
                cursor.execute("""
                    SELECT id, category, title, content, keywords, vector
                    FROM documents
                """)
            
            results = []
            
            for row in cursor.fetchall():
                doc_id, category, title, content, keywords_str, vector_str = row
                
                # parse vector
                doc_vector = self._string_to_vector(vector_str)
                
                # hitung similarity
                similarity = self._cosine_similarity(query_vector, doc_vector)
                
                if similarity > 0.1:  # threshold
                    results.append({
                        'id': doc_id,
                        'category': category,
                        'title': title,
                        'content': content,
                        'keywords': json.loads(keywords_str) if keywords_str else [],
                        'similarity_score': similarity
                    })
            
            # sort by similarity
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ error searching: {e}")
            return []
    
    def build_context(self, query: str, top_k: int = 3, category_filter: Optional[str] = None) -> str:
        """build context untuk prompt"""
        results = self.search(query, top_k, category_filter)
        
        if not results:
            return ""
        
        context_parts = []
        
        for result in results:
            score = result['similarity_score']
            content = result['content']
            category = result['category']
            
            # trim content berdasarkan relevancy
            if score > 0.7:
                max_len = 300
            elif score > 0.4:
                max_len = 200
            else:
                max_len = 100
            
            if len(content) > max_len:
                content = content[:max_len] + "..."
            
            context_parts.append(f"[{category}] {content}")
        
        return "\n".join(context_parts)
    
    def suggest_topics(self, query: str, limit: int = 3) -> List[str]:
        """suggest related topics"""
        results = self.search(query, top_k=5)
        
        topics = []
        seen_titles = set()
        
        for result in results:
            if result['similarity_score'] > 0.3:
                title = result['title']
                if title and title not in seen_titles:
                    # clean title
                    clean_title = re.sub(r'^(keahlian|pengalaman|proyek|hobi)\s+', '', title.lower())
                    clean_title = clean_title.title()
                    topics.append(clean_title)
                    seen_titles.add(title)
        
        return topics[:limit]
    
    def get_stats(self) -> Dict:
        """get storage statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM vocabulary")
        vocab_size = cursor.fetchone()[0]
        
        # get db file size
        import os
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        return {
            'document_count': doc_count,
            'vocabulary_size': vocab_size,
            'database_size_mb': round(db_size / 1024 / 1024, 2),
            'index_type': 'TF-IDF',
            'storage_type': 'SQLite'
        }
    
    def close(self):
        """tutup koneksi database"""
        if self.conn:
            self.conn.close()


# helper function untuk inisialisasi
def initialize_sqlite_rag(knowledge_data: List[Dict]) -> SQLiteVectorStorage:
    """initialize sqlite vector storage dengan knowledge data"""
    try:
        storage = SQLiteVectorStorage()
        
        if knowledge_data:
            success = storage.add_documents(knowledge_data)
            if success:
                stats = storage.get_stats()
                logger.info(f"✅ sqlite rag initialized: {stats}")
                return storage
        
        return storage
        
    except Exception as e:
        logger.error(f"❌ failed to initialize sqlite rag: {e}")
        return None