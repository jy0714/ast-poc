"""헬스체크 엔드포인트"""

from fastapi import APIRouter

from src.utils.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": "0.1.0",
        "security_mode": settings.security_mode,
        "llm_model": settings.ollama_llm_model,
    }
