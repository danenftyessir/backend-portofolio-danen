import json
import os
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
import openai
from sentence_transformers import SentenceTransformer
import requests

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, openai_api_key: str = None, use_local_embeddings: bool = True):
        """
        inisialisasi rag system dengan chromadb dan embedding model
        
        args:
            openai_api_key: api key untuk openai embeddings (optional jika pakai local)
            use_local_embeddings: gunakan sentence-transformers lokal atau openai api
        """
        self.openai_api_key = openai_api_key
        self.use_local_embeddings = use_local_embeddings
        
        # setup chromadb with new configuration
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            logger.info("chromadb client initialized successfully")
        except Exception as e:
            logger.error(f"failed to initialize chromadb: {e}")
            raise
        
        # setup embedding model
        if use_local_embeddings:
            try:
                # pakai model yang ringan dan bagus untuk bahasa indonesia
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("local embedding model loaded successfully")
            except Exception as e:
                logger.error(f"failed to load local embedding model: {e}")
                raise
        else:
            if not openai_api_key:
                raise ValueError("openai api key required for openai embeddings")
            openai.api_key = openai_api_key
        
        # nama collection untuk portfolio
        self.collection_name = "portfolio_knowledge"
        self.collection = None
        
    def get_embedding(self, text: str) -> List[float]:
        """generate embedding untuk text"""
        try:
            if self.use_local_embeddings:
                # pakai sentence-transformers
                embedding = self.embedding_model.encode(text).tolist()
            else:
                # pakai openai api
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embedding = response['data'][0]['embedding']
            
            return embedding
        except Exception as e:
            logger.error(f"failed to generate embedding: {e}")
            raise
    
    def load_knowledge_base(self, knowledge_data: List[Dict]) -> bool:
        """load knowledge base ke chromadb"""
        try:
            # hapus collection lama jika ada
            try:
                self.chroma_client.delete_collection(self.collection_name)
                logger.info("deleted existing collection")
            except:
                pass
            
            # buat collection baru
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "portfolio knowledge base"}
            )
            
            # process dan simpan setiap dokumen
            documents = []
            embeddings = []
            ids = []
            metadatas = []
            
            for item in knowledge_data:
                # gabungkan title dan content untuk context yang lebih lengkap
                full_text = f"{item['title']}: {item['content']}"
                
                embedding = self.get_embedding(full_text)
                
                documents.append(full_text)
                embeddings.append(embedding)
                ids.append(item['id'])
                metadatas.append({
                    'title': item['title'],
                    'category': item['category']
                })
            
            # batch insert ke chromadb
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"loaded {len(knowledge_data)} documents to knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"failed to load knowledge base: {e}")
            return False
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 3, category_filter: str = None) -> List[Dict]:
        """retrieve dokumen yang relevan berdasarkan query"""
        try:
            if not self.collection:
                # coba load collection yang ada
                try:
                    self.collection = self.chroma_client.get_collection(self.collection_name)
                except:
                    logger.error("no knowledge base found. please load knowledge base first.")
                    return []
            
            # generate embedding untuk query
            query_embedding = self.get_embedding(query)
            
            # setup filter berdasarkan kategori jika ada
            where_filter = None
            if category_filter:
                where_filter = {"category": category_filter}
            
            # query chromadb
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter
            )
            
            # format hasil
            retrieved_docs = []
            if results['documents'] and len(results['documents'][0]) > 0:
                docs = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for i, doc in enumerate(docs):
                    retrieved_docs.append({
                        'content': doc,
                        'metadata': metadatas[i],
                        'similarity_score': 1 - distances[i]  # convert distance ke similarity
                    })
            
            logger.info(f"retrieved {len(retrieved_docs)} relevant documents for query: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"failed to retrieve documents: {e}")
            return []
    
    def build_rag_context(self, query: str, top_k: int = 3, category_filter: str = None) -> str:
        """build context dari retrieved documents untuk prompt"""
        retrieved_docs = self.retrieve_relevant_docs(query, top_k, category_filter)
        
        if not retrieved_docs:
            return ""
        
        # format context dengan informasi relevancy
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            similarity = doc['similarity_score']
            content = doc['content']
            category = doc['metadata'].get('category', 'general')
            
            # hanya pakai dokumen dengan similarity score yang cukup tinggi
            if similarity > 0.7:  # threshold bisa di-adjust
                context_parts.append(f"[{category}] {content}")
        
        if context_parts:
            context = "Informasi relevan dari knowledge base:\n" + "\n".join(context_parts)
            return context
        else:
            return ""
    
    def suggest_related_topics(self, query: str, top_k: int = 5) -> List[str]:
        """suggest topik terkait berdasarkan query"""
        try:
            retrieved_docs = self.retrieve_relevant_docs(query, top_k)
            topics = []
            
            for doc in retrieved_docs:
                if doc['similarity_score'] > 0.6:
                    title = doc['metadata'].get('title', '')
                    if title and title not in topics:
                        topics.append(title)
            
            return topics[:3]  # return max 3 topics
            
        except Exception as e:
            logger.error(f"failed to suggest topics: {e}")
            return []

# utility functions
def load_knowledge_from_file(file_path: str) -> List[Dict]:
    """load knowledge base dari file json"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"failed to load knowledge from file: {e}")
        return []

def initialize_rag_system(knowledge_data: List[Dict] = None, use_openai: bool = False) -> RAGSystem:
    """initialize rag system dengan knowledge base"""
    try:
        openai_key = os.getenv("OPENAI_API_KEY") if use_openai else None
        rag = RAGSystem(
            openai_api_key=openai_key,
            use_local_embeddings=not use_openai
        )
        
        if knowledge_data:
            success = rag.load_knowledge_base(knowledge_data)
            if not success:
                logger.error("failed to load knowledge base")
                return None
        
        return rag
        
    except Exception as e:
        logger.error(f"failed to initialize rag system: {e}")
        return None