"""설정 엔드포인트 (보안 모드 토글 등)"""

from fastapi import APIRouter
from pydantic import BaseModel

from src.utils.config import settings

router = APIRouter()


class SecurityModeRequest(BaseModel):
    mode: str  # "on" | "off"


class SettingsResponse(BaseModel):
    security_mode: str
    ollama_llm_model: str
    ollama_embed_model: str
    openai_model: str
    chunk_size_docs: int
    chat_window_minutes: int


@router.get("/", response_model=SettingsResponse)
async def get_settings():
    """현재 설정 조회"""
    return SettingsResponse(
        security_mode=settings.security_mode,
        ollama_llm_model=settings.ollama_llm_model,
        ollama_embed_model=settings.ollama_embed_model,
        openai_model=settings.openai_model,
        chunk_size_docs=settings.chunk_size_docs,
        chat_window_minutes=settings.chat_window_minutes,
    )


@router.put("/security-mode")
async def set_security_mode(request: SecurityModeRequest):
    """보안 모드 변경 (on/off)"""
    settings.security_mode = request.mode
    return {
        "security_mode": settings.security_mode,
        "message": f"보안 모드가 {'ON (로컬 전용)' if settings.is_secure_mode else 'OFF (외부 API 허용)'}으로 변경되었습니다.",
    }
