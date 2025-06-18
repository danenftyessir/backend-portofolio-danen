from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Optional, AsyncGenerator
import logging
import time
import uuid
import random
import json
import asyncio

from ..models import (
    ChatRequest, ChatResponse, MockChatResponse, 
    SuggestedFollowupsResponse, MessageType
)
from ..config import get_settings, Settings
from ..dependencies import (
    get_rag_service, get_session_service, get_cache_service,
    check_rate_limit, validate_session_id
)
from ..services.rag_service import RAGService
from ..services.session_service import SessionService
from ..services.cache_service import CacheService
from ..services.ai_service import AIService

logger = logging.getLogger(__name__)
router = APIRouter()

async def stream_response(text: str, delay: float = 0.03) -> AsyncGenerator[str, None]:
    """stream text response word by word"""
    words = text.split()
    for i, word in enumerate(words):
        # simulasi typing delay
        await asyncio.sleep(delay)
        
        # kirim word dengan space kecuali word terakhir
        if i < len(words) - 1:
            yield f"data: {json.dumps({'chunk': word + ' ', 'done': False})}\n\n"
        else:
            yield f"data: {json.dumps({'chunk': word, 'done': False})}\n\n"
    
    # signal completion
    yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"

@router.post("/ask")
async def ask_ai(
    request: ChatRequest,
    req: Request,
    rag_service: RAGService = Depends(get_rag_service),
    session_service: SessionService = Depends(get_session_service),
    cache_service: CacheService = Depends(get_cache_service),
    settings: Settings = Depends(get_settings),
    rate_limit_ok: bool = Depends(check_rate_limit)
):
    """main chat endpoint dengan rag system dan streaming"""
    start_time = time.time()
    
    if not rate_limit_ok:
        raise HTTPException(
            status_code=429,
            detail="terlalu banyak request, coba lagi nanti"
        )
    
    try:
        # validate dan generate session id
        session_id = validate_session_id(request.session_id)
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # update session
        await session_service.update_session(session_id, request.question)
        
        # check cache terlebih dahulu
        cached_response = await cache_service.get_response(request.question)
        if cached_response:
            logger.info(f"returning cached response for: {request.question[:50]}...")
            
            # check if client accepts streaming
            accept_header = req.headers.get("accept", "")
            if "text/event-stream" in accept_header:
                # stream cached response
                return StreamingResponse(
                    stream_response(cached_response["response"]),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Session-ID": session_id,
                        "X-Message-Type": cached_response.get("message_type", "general"),
                        "X-From-Cache": "true"
                    }
                )
            else:
                # regular json response
                return ChatResponse(
                    response=cached_response["response"],
                    session_id=session_id,
                    message_type=MessageType(cached_response.get("message_type", "general")),
                    related_topics=cached_response.get("related_topics", []),
                    confidence_score=cached_response.get("confidence_score"),
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )
        
        # classify message type
        message_type = classify_message_type(request.question)
        
        # generate response
        if rag_service and (settings.gemini_api_key or settings.openai_api_key):
            # gunakan ai service dengan rag
            ai_service = AIService(settings)
            
            if message_type in [MessageType.PROFESSIONAL, MessageType.PERSONAL]:
                # retrieve relevant context untuk professional dan personal questions
                context = await rag_service.retrieve_context(request.question)
                response = await ai_service.generate_response(
                    question=request.question,
                    context=context,
                    message_type=message_type,
                    conversation_history=request.conversation_history
                )
                related_topics = await rag_service.get_related_topics(request.question)
            else:
                # untuk greeting, feedback - langsung generate
                response = await ai_service.generate_response(
                    question=request.question,
                    context="",
                    message_type=message_type,
                    conversation_history=request.conversation_history
                )
                related_topics = []
            
            confidence_score = 0.9
        else:
            # fallback ke mock response
            response = generate_mock_response(request.question, message_type)
            related_topics = []
            confidence_score = 0.5
        
        # cache response
        await cache_service.cache_response(
            question=request.question,
            response=response,
            message_type=message_type.value,
            related_topics=related_topics,
            confidence_score=confidence_score
        )
        
        # save conversation
        await session_service.add_conversation_item(
            session_id=session_id,
            question=request.question,
            response=response,
            message_type=message_type.value,
            confidence_score=confidence_score
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # check if client accepts streaming
        accept_header = req.headers.get("accept", "")
        if "text/event-stream" in accept_header:
            # stream response
            return StreamingResponse(
                stream_response(response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Session-ID": session_id,
                    "X-Message-Type": message_type.value,
                    "X-Processing-Time": str(processing_time),
                    "X-From-Cache": "false"
                }
            )
        else:
            # regular json response
            return ChatResponse(
                response=response,
                session_id=session_id,
                message_type=message_type,
                related_topics=related_topics,
                confidence_score=confidence_score,
                processing_time_ms=processing_time
            )
        
    except Exception as e:
        logger.error(f"error in ask endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"terjadi kesalahan: {str(e)}"
        )

@router.post("/ask-mock", response_model=MockChatResponse)
async def ask_mock(
    request: ChatRequest,
    req: Request,
    session_service: SessionService = Depends(get_session_service)
):
    """mock endpoint untuk testing/offline mode dengan streaming"""
    try:
        session_id = validate_session_id(request.session_id)
        if not session_id:
            session_id = str(uuid.uuid4())
        
        await session_service.update_session(session_id, request.question)
        
        message_type = classify_message_type(request.question)
        response = generate_mock_response(request.question, message_type)
        
        # check if client accepts streaming
        accept_header = req.headers.get("accept", "")
        if "text/event-stream" in accept_header:
            # stream mock response dengan delay lebih cepat
            return StreamingResponse(
                stream_response(response, delay=0.02),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Session-ID": session_id,
                    "X-Mock": "true"
                }
            )
        else:
            return MockChatResponse(
                response=response,
                session_id=session_id
            )
        
    except Exception as e:
        logger.error(f"error in mock endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"terjadi kesalahan: {str(e)}"
        )

@router.get("/suggested-followups/{session_id}", response_model=SuggestedFollowupsResponse)
async def get_suggested_followups(
    session_id: str,
    session_service: SessionService = Depends(get_session_service)
):
    """mendapatkan suggested followup questions"""
    try:
        session = await session_service.get_session(session_id)
        if not session:
            # return default followups
            followups = [
                "ceritakan lebih detail tentang pengalaman python kamu",
                "apa proyek paling challenging yang pernah kamu kerjakan?",
                "bagaimana cara kamu mengatasi technical challenges?"
            ]
        else:
            # generate followups based on conversation context
            followups = generate_contextual_followups(session)
        
        return SuggestedFollowupsResponse(
            suggested_followups=followups[:3],  # maksimal 3
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"error getting followups: {e}")
        return SuggestedFollowupsResponse(
            suggested_followups=[],
            session_id=session_id
        )

def classify_message_type(question: str) -> MessageType:
    """classify tipe pesan berdasarkan pertanyaan"""
    question_lower = question.lower()
    
    # greeting patterns
    greeting_keywords = ["halo", "hai", "hello", "hi", "selamat", "assalamualaikum"]
    if any(keyword in question_lower for keyword in greeting_keywords):
        return MessageType.GREETING
    
    # professional patterns
    professional_keywords = [
        "skill", "keahlian", "experience", "pengalaman", "project", "proyek",
        "teknologi", "algoritma", "python", "java", "web development", "data science",
        "hire", "rekrut", "interview", "kerja", "karir", "challenging", "solver"
    ]
    if any(keyword in question_lower for keyword in professional_keywords):
        return MessageType.PROFESSIONAL
    
    # personal patterns
    personal_keywords = [
        "hobi", "suka", "favorit", "musik", "lagu", "makanan", "buku",
        "novel", "travel", "wisata", "kuliner", "street food", "makan"
    ]
    if any(keyword in question_lower for keyword in personal_keywords):
        return MessageType.PERSONAL
    
    # feedback patterns
    feedback_keywords = ["terima kasih", "thanks", "bagus", "helpful", "membantu"]
    if any(keyword in question_lower for keyword in feedback_keywords):
        return MessageType.FEEDBACK
    
    return MessageType.GENERAL

def generate_mock_response(question: str, message_type: MessageType) -> str:
    """generate mock response untuk offline mode"""
    
    question_lower = question.lower()
    
    if message_type == MessageType.GREETING:
        responses = [
            "halo! saya danendra, senang bertemu dengan anda. ada yang bisa saya bantu tentang pengalaman dan proyek saya?",
            "hi! terima kasih sudah mengunjungi portfolio saya. silakan tanyakan apa saja tentang background teknis atau personal saya.",
            "selamat datang! saya siap menjawab pertanyaan tentang keahlian programming, proyek yang pernah dikerjakan, atau hal personal lainnya."
        ]
    elif message_type == MessageType.PROFESSIONAL:
        if "python" in question_lower:
            return """python adalah bahasa utama saya untuk analisis data dan machine learning. saya menguasai pandas untuk data manipulation, scikit-learn untuk machine learning models, matplotlib dan seaborn untuk visualization, dan numpy untuk numerical computing. 

pengalaman 1 tahun khusus di data science dengan berbagai project algoritma kompleks seperti rush hour puzzle solver yang mengimplementasikan multiple pathfinding algorithms. python juga saya gunakan untuk backend development dengan fastapi, seperti yang terlihat di portfolio ini.

yang paling saya suka dari python adalah versatility-nya - bisa untuk web development, data science, automation, sampai ai development. ecosystem library-nya juga sangat rich dan community support yang luar biasa."""
        elif "challenging" in question_lower or "solver" in question_lower:
            return """rush hour puzzle solver adalah project yang paling technically challenging dan educational. saya implement multiple pathfinding algorithms - ucs untuk optimal solutions, greedy best-first untuk speed, a* untuk balanced approach, dan dijkstra untuk comprehensive exploration.

biggest challenge adalah optimizing algorithm performance untuk handle complex puzzle configurations. saya develop custom heuristic functions dan implement efficient state representation untuk minimize memory usage. plus, created interactive visualization yang allow users untuk understand algorithm behavior step-by-step.

project ini ngajarin saya banyak tentang algorithm optimization, memory management, dan user experience design. complexity analysis juga jadi lebih mendalam karena harus compare performance antar algoritma."""
        else:
            responses = [
                "saya memiliki pengalaman 2 tahun dalam web development dan 1 tahun fokus di data science. keahlian utama meliputi python, java, next.js, dan algoritma kompleks.",
                "proyek favorit saya adalah rush hour puzzle solver yang mengimplementasikan multiple pathfinding algorithms seperti a*, dijkstra, dan ucs dengan optimasi performa tinggi.",
                "sebagai mahasiswa teknik informatika itb semester 4, saya aktif mengembangkan skills di bidang algoritma dan data science melalui berbagai project challenging."
            ]
    elif message_type == MessageType.PERSONAL:
        if "makanan" in question_lower or "makan" in question_lower or "favorit" in question_lower:
            return """untuk makanan, saya obsessed sama street food indonesia! martabak manis jadi comfort food utama - yang paling suka varian coklat keju dengan topping kacang. sate ayam juga favorit banget, terutama yang dari abang-abang kaki lima dengan bumbu kacang yang kental.

jakarta punya spot-spot legendary kayak sabang dan pecenongan yang classic banget buat late night food hunting. yang bikin saya suka street food bukan cuma rasanya, tapi whole experience-nya - social interaction dengan penjual, atmosphere di pinggir jalan, dan feeling nostalgic yang ga bisa didapat di restoran fancy.

gorengan juga weakness saya, terutama pisang goreng dan tempe mendoan pas hujan-hujan. ketoprak dan batagor juga masuk list favorit. street food culture indonesia itu rich banget dan setiap daerah punya signature dishes yang unik."""
        elif "musik" in question_lower or "lagu" in question_lower:
            return """selera musik saya nostalgic & oldies. lagi relate banget sama 'without you' air supply dan 'sekali ini saja' glenn fredly. oldies punya emotional depth dan musical complexity yang susah dicari di modern music.

lirik-liriknya meaningful, production quality tinggi, dan timeless. untuk coding biasanya pakai lo-fi beats atau soundtrack film kayak star wars yang bikin suasana lebih intens. glenn fredly special karena pioneer indonesian jazz-soul dengan voice quality yang luar biasa.

musik juga jadi companion saat problem-solving. rhythm yang steady dari oldies somehow help maintain focus selama coding marathon atau algorithm design sessions."""
        else:
            responses = [
                "hobi saya membaca novel fantasy seperti omniscient reader viewpoint, traveling ke destinasi lokal, dan hunting street food di jakarta.",
                "untuk musik, saya suka oldies seperti air supply dan glenn fredly. selera kuliner lebih ke street food seperti martabak manis dan sate ayam.",
                "saya penggemar berat street food jakarta dan novel dengan world-building yang kompleks. reading habit ini membantu analytical thinking dalam problem-solving."
            ]
    elif message_type == MessageType.FEEDBACK:
        responses = [
            "terima kasih! senang bisa membantu. jangan ragu untuk bertanya hal lain tentang pengalaman atau proyek saya.",
            "sama-sama! jika ada pertanyaan lain tentang technical skills atau background saya, silakan tanyakan kapan saja.",
            "glad to help! feel free to explore more about my projects, skills, atau aspek personal lainnya."
        ]
    else:
        responses = [
            "hmm, bisa dijelaskan lebih spesifik? saya siap membantu dengan informasi tentang pengalaman teknis, proyek, atau hal personal.",
            "pertanyaan yang menarik! silakan elaborate lebih detail agar saya bisa memberikan jawaban yang lebih tepat.",
            "saya akan coba bantu sebaik mungkin. bisa diperjelas konteks pertanyaannya? apakah terkait technical skills atau personal?"
        ]
    
    # jika responses masih list, pilih random
    if isinstance(responses, list):
        return random.choice(responses)
    else:
        return responses

def extract_related_topics(context: str) -> list:
    """extract related topics dari retrieved context"""
    # simple topic extraction berdasarkan keywords
    topics = set()
    
    context_lower = context.lower()
    
    # technical topics
    if "python" in context_lower:
        topics.add("keahlian python")
    if "rush hour" in context_lower:
        topics.add("rush hour puzzle solver")
    if "web development" in context_lower or "next.js" in context_lower:
        topics.add("web development")
    if "street food" in context_lower or "kuliner" in context_lower:
        topics.add("street food enthusiast")
    if "musik" in context_lower or "lagu" in context_lower:
        topics.add("selera musik")
    
    return list(topics)[:5]  # maksimal 5 topics

def generate_contextual_followups(session) -> list:
    """generate followup questions berdasarkan session context"""
    base_followups = [
        "ceritakan lebih detail tentang rush hour solver project",
        "apa challenge terbesar dalam mengembangkan algoritma pencarian?",
        "bagaimana pengalaman jadi asisten praktikum di itb?",
        "teknologi apa yang ingin dipelajari selanjutnya?",
        "rekomendasi street food favorit di jakarta dong"
    ]
    
    # bisa dikembangkan untuk analyze conversation history
    # dan generate more contextual followups
    
    return random.sample(base_followups, min(3, len(base_followups)))