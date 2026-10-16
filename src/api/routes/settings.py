"""설정 엔드포인트 (보안 모드 토글, 임베딩 모델 관리 등)"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class SecurityModeRequest(BaseModel):
    mode: str  # "on" | "off"


class EmbedModelRequest(BaseModel):
    model: str  # e.g. "bge-m3", "nomic-embed-text"


class RerankModeRequest(BaseModel):
    enabled: bool


class SettingsResponse(BaseModel):
    security_mode: str
    ollama_llm_model: str
    ollama_embed_model: str
    openai_model: str
    unit_max_part_tokens: int
    chat_session_gap_minutes: int
    rerank_enabled: bool


class EmbedModelResponse(BaseModel):
    model: str
    dimension: int | None = None
    reindex_required: list[str] = []
    message: str = ""


@router.get("/", response_model=SettingsResponse)
async def get_settings():
    """현재 설정 조회"""
    return SettingsResponse(
        security_mode=settings.security_mode,
        ollama_llm_model=settings.ollama_llm_model,
        ollama_embed_model=settings.ollama_embed_model,
        openai_model=settings.openai_model,
        unit_max_part_tokens=settings.unit_max_part_tokens,
        chat_session_gap_minutes=settings.chat_session_gap_minutes,
        rerank_enabled=settings.rerank_enabled,
    )


@router.put("/security-mode")
async def set_security_mode(request: SecurityModeRequest):
    """보안 모드 변경 (on/off)"""
    settings.security_mode = request.mode
    return {
        "security_mode": settings.security_mode,
        "message": f"보안 모드가 {'ON (로컬 전용)' if settings.is_secure_mode else 'OFF (외부 API 허용)'}으로 변경되었습니다.",
    }


@router.put("/rerank-mode")
async def set_rerank_mode(request: RerankModeRequest):
    """Reranker ON/OFF 전역 토글"""
    settings.rerank_enabled = request.enabled
    status = "ON" if settings.rerank_enabled else "OFF"
    logger.info(f"Reranker 모드 변경: {status}")
    return {
        "rerank_enabled": settings.rerank_enabled,
        "rerank_model": settings.rerank_model,
        "message": f"Reranker가 {status}으로 변경되었습니다.",
    }



def _cases_needing_reindex(model: str, dim: int | None) -> list[str]:
    """서비스 프로필 색인이 이 임베딩 모델·차원과 다른 READY/ARCHIVED 케이스 (v4 색인이 없는 케이스 포함)

    Chroma 컬렉션 대신 케이스별 프로필 meta.json으로 판단한다(지시서 §5.7).
    """
    from src.cases.case_store import CaseStatus, CaseStore
    from src.index import case_index

    out: list[str] = []
    try:
        cases = CaseStore().list_all()
    except Exception as e:
        logger.warning(f"케이스 호환성 확인 실패 (무시): {e}")
        return out
    for case in cases:
        if case.status not in (CaseStatus.READY, CaseStatus.ARCHIVED):
            continue
        try:
            meta = case_index.load(case.case_id).profile_meta(settings.service_index_profile)
        except Exception:
            out.append(case.case_id)  # v4 색인 없음·버전 불일치 → 재인덱싱 필요
            continue
        if meta.get("embed_model") != model or (dim is not None and meta.get("embed_dim") != dim):
            out.append(case.case_id)
    return out


@router.put("/embed-model", response_model=EmbedModelResponse)
async def set_embed_model(request: EmbedModelRequest):
    """임베딩 모델 변경 + 재인덱싱 필요 케이스 경고

    모델 변경 시 벡터 차원이 달라지면 기존 인덱싱된 케이스와
    호환되지 않으므로 재인덱싱이 필요합니다.
    """
    from src.embeddings.embedding_service import KNOWN_DIMENSIONS, EmbeddingService

    old_model = settings.ollama_embed_model
    new_model = request.model.strip()

    if not new_model:
        return EmbedModelResponse(
            model=old_model,
            message="모델명이 비어 있습니다.",
        )

    # 새 모델의 차원 확인
    new_service = EmbeddingService(model=new_model)
    try:
        new_dim = new_service.get_dimension()
    except ConnectionError:
        # Ollama에 없는 모델이면 알려진 차원만 확인
        new_dim = KNOWN_DIMENSIONS.get(new_model)

    # 케이스 프로필 색인과 비교 → 재인덱싱 필요 케이스 수집
    reindex_required = _cases_needing_reindex(new_model, new_dim)

    # 설정 변경 적용
    settings.ollama_embed_model = new_model

    # 메시지 생성
    if old_model == new_model:
        message = f"임베딩 모델이 이미 '{new_model}'입니다."
    elif reindex_required:
        message = (
            f"임베딩 모델이 '{old_model}' → '{new_model}'로 변경되었습니다. "
            f"⚠️ 색인 모델·차원이 달라 {len(reindex_required)}개 케이스의 재인덱싱이 필요합니다: "
            f"{', '.join(reindex_required)}"
        )
    else:
        message = f"임베딩 모델이 '{old_model}' → '{new_model}'로 변경되었습니다."

    logger.info(f"임베딩 모델 변경: {old_model} → {new_model} (dim={new_dim})")

    return EmbedModelResponse(
        model=new_model,
        dimension=new_dim,
        reindex_required=reindex_required,
        message=message,
    )


@router.get("/embed-model", response_model=EmbedModelResponse)
async def get_embed_model():
    """현재 임베딩 모델 정보 + 케이스 호환성 확인"""
    from src.embeddings.embedding_service import KNOWN_DIMENSIONS, EmbeddingService

    model = settings.ollama_embed_model
    service = EmbeddingService(model=model)

    try:
        dim = service.get_dimension()
    except ConnectionError:
        dim = KNOWN_DIMENSIONS.get(model)

    # 호환성 확인 (케이스 프로필 색인 기준)
    mismatched = _cases_needing_reindex(model, dim)

    message = f"현재 모델: {model} ({dim}차원)"
    if mismatched:
        message += f" — ⚠️ {len(mismatched)}개 케이스 색인 모델 불일치, 재인덱싱 필요"

    return EmbedModelResponse(
        model=model,
        dimension=dim,
        reindex_required=mismatched,
        message=message,
    )
