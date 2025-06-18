import json
import re
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
import difflib

logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    """enhanced simple rag system dengan full functionality"""
    
    def __init__(self):
        self.documents: List[Dict] = []
        self.keyword_index: Dict[str, List[int]] = defaultdict(list)
        self.category_index: Dict[str, List[int]] = defaultdict(list)
        self.title_index: Dict[str, int] = {}
        self.content_vectors: List[Dict[str, float]] = []
        
    def load_knowledge_base(self, knowledge_data: List[Dict]) -> bool:
        """load dan index knowledge base"""
        try:
            self.documents = knowledge_data
            self._build_advanced_indexes()
            logger.info(f"✅ simple rag loaded {len(knowledge_data)} documents")
            return True
        except Exception as e:
            logger.error(f"❌ error loading knowledge base: {e}")
            return False
    
    def _build_advanced_indexes(self):
        """build comprehensive indexes untuk efficient retrieval"""
        doc_freq = defaultdict(int)
        all_words = set()
        
        # first pass: collect all words dan document frequencies
        for i, doc in enumerate(self.documents):
            category = doc.get('category', 'general')
            self.category_index[category].append(i)
            
            # index by title untuk exact matching
            title = doc.get('title', '').lower()
            if title:
                self.title_index[title] = i
            
            # extract words dari content, title, dan keywords
            content = doc.get('content', '').lower()
            keywords = doc.get('keywords', [])
            
            # comprehensive word extraction
            text_content = f"{content} {title} {' '.join(keywords)}"
            words = re.findall(r'\b\w+\b', text_content.lower())
            
            # filter words dan build vocabulary
            meaningful_words = [w for w in words if len(w) > 2 and not w.isdigit()]
            doc_words = set(meaningful_words)
            
            # update document frequency
            for word in doc_words:
                doc_freq[word] += 1
                all_words.add(word)
            
            # build word index untuk fast lookup
            for word in meaningful_words:
                if i not in self.keyword_index[word]:
                    self.keyword_index[word].append(i)
        
        # second pass: build tf-idf style vectors
        total_docs = len(self.documents)
        for i, doc in enumerate(self.documents):
            content = doc.get('content', '').lower()
            title = doc.get('title', '').lower()
            keywords = doc.get('keywords', [])
            
            text_content = f"{content} {title} {' '.join(keywords)}"
            words = re.findall(r'\b\w+\b', text_content.lower())
            meaningful_words = [w for w in words if len(w) > 2 and not w.isdigit()]
            
            # calculate term frequencies
            word_count = defaultdict(int)
            for word in meaningful_words:
                word_count[word] += 1
            
            # build tf-idf style vector
            doc_vector = {}
            for word, count in word_count.items():
                tf = count / len(meaningful_words) if meaningful_words else 0
                idf = total_docs / doc_freq[word] if doc_freq[word] > 0 else 0
                
                # weight boost untuk keywords dan title
                weight_multiplier = 1.0
                if word in [kw.lower() for kw in keywords]:
                    weight_multiplier = 3.0
                elif word in title:
                    weight_multiplier = 2.0
                
                doc_vector[word] = tf * idf * weight_multiplier
            
            self.content_vectors.append(doc_vector)
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 3, category_filter: str = None) -> List[Dict]:
        """advanced retrieval dengan multiple scoring methods"""
        try:
            query_words = re.findall(r'\b\w+\b', query.lower())
            query_words = [w for w in query_words if len(w) > 2]
            
            if not query_words:
                return []
            
            doc_scores = defaultdict(float)
            
            # method 1: exact keyword matching dengan boosting
            for word in query_words:
                if word in self.keyword_index:
                    for doc_idx in self.keyword_index[word]:
                        # filter by category kalau ada
                        if category_filter:
                            doc_category = self.documents[doc_idx].get('category', '')
                            if doc_category != category_filter:
                                continue
                        
                        # scoring berdasarkan context
                        doc = self.documents[doc_idx]
                        keywords = [kw.lower() for kw in doc.get('keywords', [])]
                        title = doc.get('title', '').lower()
                        content = doc.get('content', '').lower()
                        
                        # progressive scoring
                        if word in keywords:
                            doc_scores[doc_idx] += 5.0
                        elif word in title:
                            doc_scores[doc_idx] += 3.0
                        elif word in content:
                            doc_scores[doc_idx] += 1.0
                
                # method 2: fuzzy matching untuk typos
                similar_words = difflib.get_close_matches(
                    word, self.keyword_index.keys(), n=3, cutoff=0.8
                )
                for similar_word in similar_words:
                    if similar_word != word:
                        for doc_idx in self.keyword_index[similar_word]:
                            if category_filter:
                                doc_category = self.documents[doc_idx].get('category', '')
                                if doc_category != category_filter:
                                    continue
                            doc_scores[doc_idx] += 0.5
            
            # method 3: tf-idf style similarity
            for doc_idx, doc_vector in enumerate(self.content_vectors):
                if category_filter:
                    doc_category = self.documents[doc_idx].get('category', '')
                    if doc_category != category_filter:
                        continue
                
                # calculate cosine similarity dengan query
                query_vector_sum = 0
                doc_vector_sum = 0
                dot_product = 0
                
                for word in query_words:
                    if word in doc_vector:
                        word_weight = doc_vector[word]
                        dot_product += word_weight
                        doc_vector_sum += word_weight ** 2
                        query_vector_sum += 1
                
                if doc_vector_sum > 0 and query_vector_sum > 0:
                    similarity = dot_product / (doc_vector_sum ** 0.5 * query_vector_sum ** 0.5)
                    doc_scores[doc_idx] += similarity * 2.0
            
            # sort by score dan return top_k
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            results = []
            for doc_idx, score in sorted_docs[:top_k]:
                if score > 0.1:
                    doc = self.documents[doc_idx]
                    results.append({
                        'content': f"{doc['title']}: {doc['content']}",
                        'metadata': {
                            'title': doc['title'],
                            'category': doc['category'],
                            'keywords': doc.get('keywords', [])
                        },
                        'similarity_score': min(score / 10.0, 1.0)
                    })
            
            logger.info(f"retrieved {len(results)} docs for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"❌ error retrieving docs: {e}")
            return []
    
    def build_rag_context(self, query: str, top_k: int = 3, category_filter: str = None) -> str:
        """build comprehensive context dari retrieved docs"""
        docs = self.retrieve_relevant_docs(query, top_k, category_filter)
        
        if not docs:
            return ""
        
        context_parts = []
        for doc in docs:
            if doc['similarity_score'] > 0.2:
                content = doc['content']
                category = doc['metadata']['category']
                
                # intelligent content trimming berdasarkan relevancy
                if doc['similarity_score'] > 0.7:
                    trimmed_content = content[:300] + "..." if len(content) > 300 else content
                else:
                    trimmed_content = content[:150] + "..." if len(content) > 150 else content
                
                context_parts.append(f"[{category}] {trimmed_content}")
        
        if context_parts:
            full_context = "\n".join(context_parts)
            
            # ensure total context tidak terlalu panjang
            if len(full_context) > 800:
                priority_parts = []
                current_length = 0
                for part in context_parts:
                    if current_length + len(part) <= 800:
                        priority_parts.append(part)
                        current_length += len(part)
                    else:
                        break
                full_context = "\n".join(priority_parts)
            
            return full_context
        
        return ""
    
    def suggest_related_topics(self, query: str, top_k: int = 5) -> List[str]:
        """suggest topik terkait dengan intelligent topic discovery"""
        try:
            docs = self.retrieve_relevant_docs(query, top_k)
            topics = []
            topic_scores = defaultdict(float)
            
            # collect topics dengan scoring
            for doc in docs:
                if doc['similarity_score'] > 0.3:
                    title = doc['metadata'].get('title', '').strip()
                    
                    if title:
                        # smart title processing
                        clean_title = re.sub(r'^(keahlian|pengalaman|proyek|hobi)\s+', '', title.lower())
                        clean_title = clean_title.title()
                        
                        topic_scores[clean_title] += doc['similarity_score']
            
            # sort topics by score
            sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
            
            for topic, score in sorted_topics[:3]:
                if score > 0.4 and topic not in topics:
                    topics.append(topic)
            
            # fallback topics kalau kurang
            if len(topics) < 3:
                category_topics = {
                    'keahlian': ['Keahlian Python', 'Web Development', 'Data Science'],
                    'proyek': ['Rush Hour Solver', 'Algorithm Projects', 'Portfolio Development'],
                    'hobi': ['Street Food Journey', 'Reading Habits', 'Technology Interests'],
                    'musik': ['Music Preferences', 'Coding Playlist', 'Nostalgic Songs']
                }
                
                query_lower = query.lower()
                for category, fallback_topics in category_topics.items():
                    if any(keyword in query_lower for keyword in [category, category.replace('_', ' ')]):
                        for fallback_topic in fallback_topics:
                            if fallback_topic not in topics and len(topics) < 3:
                                topics.append(fallback_topic)
                        break
            
            return topics[:3]
            
        except Exception as e:
            logger.error(f"❌ error suggesting topics: {e}")
            return []

def load_knowledge_from_file(file_path: str) -> List[Dict]:
    """load knowledge dari json file dengan validation"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # validate structure
        required_fields = ['id', 'category', 'title', 'content']
        valid_docs = []
        
        for doc in data:
            if all(field in doc for field in required_fields):
                valid_docs.append(doc)
            else:
                logger.warning(f"invalid document structure: {doc.get('id', 'unknown')}")
        
        logger.info(f"loaded {len(valid_docs)} valid documents from {file_path}")
        return valid_docs
        
    except Exception as e:
        logger.error(f"❌ error loading file: {e}")
        return []

def initialize_rag_system(knowledge_data: List[Dict] = None, use_openai: bool = False) -> Optional[SimpleRAGSystem]:
    """initialize simple rag system"""
    try:
        rag = SimpleRAGSystem()
        
        if knowledge_data:
            if rag.load_knowledge_base(knowledge_data):
                logger.info("✅ simple rag system initialized successfully")
                return rag
            else:
                logger.error("❌ failed to load knowledge base")
                return None
        else:
            logger.warning("⚠️ no knowledge data provided")
            return rag
            
    except Exception as e:
        logger.error(f"❌ error initializing rag: {e}")
        return None