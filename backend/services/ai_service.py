import logging
from typing import List, Dict, Optional
import asyncio

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

from ..config import Settings
from ..models import MessageType

logger = logging.getLogger(__name__)

class AIService:
    """service untuk integrasi dengan gemini dan openai"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gemini_client = None
        self.openai_client = None
        
        # initialize gemini client
        if GEMINI_AVAILABLE and settings.gemini_api_key:
            try:
                genai.configure(api_key=settings.gemini_api_key)
                self.gemini_client = genai.GenerativeModel(settings.gemini_model)
                logger.info("✅ gemini client initialized")
            except Exception as e:
                logger.error(f"❌ failed to initialize gemini: {e}")
        
        # initialize openai client (fallback)
        if OPENAI_AVAILABLE and settings.openai_api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
                logger.info("✅ openai client initialized (fallback)")
            except Exception as e:
                logger.error(f"❌ failed to initialize openai: {e}")
    
    async def generate_response(
        self,
        question: str,
        context: str = "",
        message_type: MessageType = MessageType.GENERAL,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """generate response menggunakan ai provider dengan fallback"""
        
        # coba gemini dulu (primary)
        if self.settings.ai_provider == "gemini" and self.gemini_client:
            try:
                response = await self._generate_with_gemini(
                    question, context, message_type, conversation_history
                )
                if response:
                    return response
                logger.warning("⚠️ gemini returned empty response, trying fallback")
            except Exception as e:
                logger.error(f"❌ gemini generation failed: {e}")
                logger.info("🔄 falling back to openai...")
        
        # fallback ke openai
        if self.openai_client:
            try:
                response = await self._generate_with_openai(
                    question, context, message_type, conversation_history
                )
                if response:
                    return response
            except Exception as e:
                logger.error(f"❌ openai generation failed: {e}")
        
        # fallback terakhir ke mock response
        logger.warning("🔄 all ai providers failed, using fallback response")
        return self._get_fallback_response(question, message_type)
    
    async def _generate_with_gemini(
        self,
        question: str,
        context: str,
        message_type: MessageType,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """generate response menggunakan gemini"""
        
        try:
            # build system prompt
            system_prompt = self._build_system_prompt(message_type)
            
            # build conversation context
            conversation_context = ""
            if conversation_history:
                for item in conversation_history[-3:]:  # ambil 3 terakhir
                    if "question" in item and "response" in item:
                        conversation_context += f"User: {item['question']}\nAssistant: {item['response']}\n\n"
            
            # build full prompt
            full_prompt = f"""{system_prompt}

{f"Informasi relevan dari knowledge base:\n{context}\n" if context else ""}

{f"Konteks percakapan sebelumnya:\n{conversation_context}" if conversation_context else ""}

Pertanyaan user: {question}

Jawab dengan natural dan sesuai personality yang telah dijelaskan:"""
            
            # call gemini api
            def sync_call():
                response = self.gemini_client.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=self.settings.max_tokens,
                        temperature=self.settings.temperature,
                        top_p=0.9,
                        top_k=40
                    )
                )
                return response.text.strip()
            
            # run in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, sync_call)
            
            logger.info("✅ gemini response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"❌ gemini api call failed: {e}")
            raise
    
    async def _generate_with_openai(
        self,
        question: str,
        context: str,
        message_type: MessageType,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """generate response menggunakan openai"""
        
        try:
            # build system prompt
            system_prompt = self._build_system_prompt(message_type)
            
            # build messages
            messages = [{"role": "system", "content": system_prompt}]
            
            # add conversation history
            if conversation_history:
                for item in conversation_history[-3:]:
                    if "question" in item and "response" in item:
                        messages.append({"role": "user", "content": item["question"]})
                        messages.append({"role": "assistant", "content": item["response"]})
            
            # add context dan current question
            if context:
                context_message = f"informasi relevan dari knowledge base:\n{context}\n\npertanyaan user:"
                messages.append({"role": "user", "content": f"{context_message} {question}"})
            else:
                messages.append({"role": "user", "content": question})
            
            # call openai api
            def sync_call():
                return self.openai_client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=messages,
                    max_tokens=self.settings.max_tokens,
                    temperature=self.settings.temperature,
                    timeout=30
                )
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, sync_call)
            
            logger.info("✅ openai response generated successfully")
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ openai api call failed: {e}")
            raise
    
    def _build_system_prompt(self, message_type: MessageType) -> str:
        """build system prompt berdasarkan message type"""
        
        base_prompt = """kamu adalah ai assistant untuk danendra shafi athallah, mahasiswa teknik informatika itb semester 4 yang passionate di bidang data science dan algoritma.

personality: friendly, technical tapi approachable, suka sharing knowledge, humble tapi confident tentang skills.

guidelines:
- jawab dalam bahasa indonesia yang natural dan conversational
- berikan jawaban yang detailed dan informatif (2-3 paragraf minimum)
- kalau ada informasi dari knowledge base, gunakan itu sebagai referensi utama dan expand dengan detail
- jangan claim hal yang tidak ada di knowledge base
- kalau tidak tahu, bilang dengan jujur dan suggest pertanyaan alternatif
- showcase enthusiasm tentang technology dan learning
- berikan context dan background untuk setiap jawaban
- gunakan storytelling approach untuk membuat jawaban lebih menarik"""

        if message_type == MessageType.GREETING:
            return base_prompt + "\n\nuntuk greeting: sambut dengan warm dan friendly, perkenalkan danendra secara singkat, dan encourage user untuk bertanya lebih lanjut."
            
        elif message_type == MessageType.PROFESSIONAL:
            return base_prompt + "\n\nuntuk professional questions: fokus pada technical skills, project experience, academic achievement, dan work experience. gunakan informasi dari knowledge base untuk detail yang akurat."
            
        if message_type == MessageType.PERSONAL:
            return base_prompt + "\n\nuntuk personal questions: sharing tentang hobi, interests, music taste, food preferences dengan storytelling approach yang engaging. berikan context mengapa suka hal tersebut, pengalaman personal, dan detail yang membuat jawaban lebih menarik. untuk makanan, ceritakan tentang street food culture, tempat favorit, dan pengalaman kuliner."
            
        elif message_type == MessageType.FEEDBACK:
            return base_prompt + "\n\nuntuk feedback: appreciate input user, respond gracefully, dan encourage further interaction."
            
        else:
            return base_prompt + "\n\nuntuk general questions: analyze pertanyaan dan respond appropriately. jika pertanyaan tidak jelas, ask for clarification dengan friendly manner."
    
    def _get_fallback_response(self, question: str, message_type: MessageType) -> str:
        """fallback response jika semua ai provider gagal"""
        
        if message_type == MessageType.GREETING:
            return "halo! saya danendra, mahasiswa teknik informatika itb yang passionate di bidang data science dan algoritma. ada yang bisa saya bantu tentang pengalaman atau proyek saya?"
            
        elif message_type == MessageType.PROFESSIONAL:
            return "maaf, saat ini saya mengalami kendala teknis dalam mengakses detail lengkap. secara umum, saya memiliki pengalaman 2 tahun web development dan 1 tahun data science dengan keahlian python, java, dan next.js. silakan coba pertanyaan yang lebih spesifik."
            
        elif message_type == MessageType.PERSONAL:
            return """untuk makanan, saya obsessed sama street food indonesia! martabak manis jadi comfort food utama - yang paling suka varian coklat keju dengan topping kacang. sate ayam juga favorit banget, terutama yang dari abang-abang kaki lima dengan bumbu kacang yang kental. 

jakarta punya spot-spot legendary kayak sabang dan pecenongan yang classic banget buat late night food hunting. yang bikin saya suka street food bukan cuma rasanya, tapi whole experience-nya - social interaction dengan penjual, atmosphere di pinggir jalan, dan feeling nostalgic yang ga bisa didapat di restoran fancy.

gorengan juga weakness saya, terutama pisang goreng dan tempe mendoan pas hujan-hujan. ketoprak dan batagor juga masuk list favorit. street food culture indonesia itu rich banget dan setiap daerah punya signature dishes yang unik."""
            
        elif message_type == MessageType.FEEDBACK:
            return "terima kasih untuk feedbacknya! senang bisa membantu. jangan ragu untuk bertanya hal lain ya."
            
        else:
            return "maaf, saya mengalami kendala teknis saat ini. bisa coba pertanyaan yang lebih spesifik tentang pengalaman teknis, proyek, atau hal personal saya?"
    
    async def generate_followup_questions(
        self,
        conversation_context: str,
        last_response: str
    ) -> List[str]:
        """generate contextual followup questions"""
        
        # coba dengan gemini dulu
        if self.gemini_client:
            try:
                prompt = f"""berdasarkan konteks percakapan berikut, generate 3 pertanyaan followup yang relevan dan natural:

konteks: {conversation_context}
jawaban terakhir: {last_response}

generate 3 pertanyaan yang:
1. relevan dengan topik yang sedang dibahas
2. natural dan conversational
3. help user explore lebih dalam tentang danendra

format: return hanya list pertanyaan, satu per baris, tanpa numbering."""

                def sync_call():
                    response = self.gemini_client.generate_content(prompt)
                    return response.text.strip()
                
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, sync_call)
                
                # parse response menjadi list
                followups = [q.strip() for q in response.split('\n') if q.strip()]
                return followups[:3]
                
            except Exception as e:
                logger.error(f"error generating followups with gemini: {e}")
        
        # fallback dengan openai
        if self.openai_client:
            try:
                # implementasi sama seperti sebelumnya
                pass
            except Exception as e:
                logger.error(f"error generating followups with openai: {e}")
        
        # fallback ke default
        return self._get_default_followups()
    
    def _get_default_followups(self) -> List[str]:
        """default followup questions"""
        return [
            "ceritakan lebih detail tentang proyek yang paling challenging",
            "apa teknologi yang paling ingin dipelajari selanjutnya?",
            "bagaimana pengalaman kuliah di itb sejauh ini?"
        ]
    
    def get_provider_status(self) -> Dict[str, bool]:
        """get status dari semua ai providers"""
        return {
            "gemini_available": bool(self.gemini_client),
            "openai_available": bool(self.openai_client),
            "primary_provider": self.settings.ai_provider,
            "gemini_library_installed": GEMINI_AVAILABLE,
            "openai_library_installed": OPENAI_AVAILABLE
        }