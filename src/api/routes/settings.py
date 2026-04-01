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


class SettingsResponse(BaseModel):
    security_mode: str
    ollama_llm_model: str
    ollama_embed_model: str
    openai_model: str
    chunk_size_docs: int
    chat_window_minutes: int


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


@router.put("/embed-model", response_model=EmbedModelResponse)
async def set_embed_model(request: EmbedModelRequest):
    """임베딩 모델 변경 + 재인덱싱 필요 케이스 경고

    모델 변경 시 벡터 차원이 달라지면 기존 인덱싱된 케이스와
    호환되지 않으므로 재인덱싱이 필요합니다.
    """
    from src.embeddings.embedding_service import KNOWN_DIMENSIONS, EmbeddingService
    from src.vectorstore.vector_store import VectorStoreService

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

    # 기존 모델과 차원 비교 → 재인덱싱 필요 케이스 수집
    reindex_required: list[str] = []

    if new_dim is not None:
        try:
            from src.cases.case_store import CaseStatus, CaseStore

            case_store = CaseStore()
            all_cases = case_store.list_all()

            for case in all_cases:
                if case.status not in (CaseStatus.READY, CaseStatus.ARCHIVED):
                    continue

                collection_name = f"case_{case.case_id}"
                try:
                    vs = VectorStoreService(collection_name=collection_name)
                    existing_dim = vs.get_collection_dimension()
                    if existing_dim is not None and existing_dim != new_dim:
                        reindex_required.append(case.case_id)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"케이스 호환성 확인 실패 (무시): {e}")

    # 설정 변경 적용
    settings.ollama_embed_model = new_model

    # 메시지 생성
    if old_model == new_model:
        message = f"임베딩 모델이 이미 '{new_model}'입니다."
    elif reindex_required:
        message = (
            f"임베딩 모델이 '{old_model}' → '{new_model}'로 변경되었습니다. "
            f"⚠️ 벡터 차원이 달라 {len(reindex_required)}개 케이스의 재인덱싱이 필요합니다: "
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
    from src.vectorstore.vector_store import VectorStoreService

    model = settings.ollama_embed_model
    service = EmbeddingService(model=model)

    try:
        dim = service.get_dimension()
    except ConnectionError:
        dim = KNOWN_DIMENSIONS.get(model)

    # 호환성 확인
    mismatched: list[str] = []
    if dim is not None:
        try:
            from src.cases.case_store import CaseStatus, CaseStore

            case_store = CaseStore()
            for case in case_store.list_all():
                if case.status not in (CaseStatus.READY, CaseStatus.ARCHIVED):
                    continue
                try:
                    vs = VectorStoreService(collection_name=f"case_{case.case_id}")
                    existing_dim = vs.get_collection_dimension()
                    if existing_dim is not None and existing_dim != dim:
                        mismatched.append(case.case_id)
                except Exception:
                    pass
        except Exception:
            pass

    message = f"현재 모델: {model} ({dim}차원)"
    if mismatched:
        message += f" — ⚠️ {len(mismatched)}개 케이스 차원 불일치, 재인덱싱 필요"

    return EmbedModelResponse(
        model=model,
        dimension=dim,
        reindex_required=mismatched,
        message=message,
    )
