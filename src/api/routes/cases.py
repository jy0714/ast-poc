"""케이스 관리 엔드포인트 (Admin UI)

CRUD + 라이프사이클:
- POST   /              — 새 케이스 생성
- GET    /              — 전체 케이스 목록
- GET    /{case_id}     — 케이스 상세 조회
- PATCH  /{case_id}     — 데이터 소스 추가
- DELETE /{case_id}     — 케이스 삭제 (벡터DB + BM25 포함)
- POST   /{case_id}/archive — 케이스 보관 처리
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.cases.case_store import (
    CaseMetadata,
    CaseNotFoundError,
    CaseStatus,
    CaseStore,
    InvalidStatusTransitionError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 싱글톤 케이스 저장소
_store = CaseStore()


def get_store() -> CaseStore:
    """케이스 저장소 인스턴스 반환 (테스트 시 교체 가능)"""
    return _store


# === 요청/응답 모델 ===


class CaseCreate(BaseModel):
    """케이스 생성 요청"""

    name: str
    description: str = ""
    pst_paths: list[str] = []
    doc_paths: list[str] = []


class CaseUpdateSources(BaseModel):
    """데이터 소스 추가 요청"""

    pst_paths: list[str] = []
    doc_paths: list[str] = []


class CaseResponse(BaseModel):
    """케이스 응답"""

    case_id: str
    name: str
    description: str
    status: str
    created_at: datetime
    updated_at: datetime
    pst_paths: list[str] = []
    doc_paths: list[str] = []
    total_documents: int = 0
    total_chunks: int = 0
    indexed_at: datetime | None = None
    error_message: str = ""


class DeleteResponse(BaseModel):
    """삭제 응답"""

    case_id: str
    message: str


def _to_response(meta: CaseMetadata) -> CaseResponse:
    """CaseMetadata → CaseResponse 변환"""
    return CaseResponse(
        case_id=meta.case_id,
        name=meta.name,
        description=meta.description,
        status=meta.status.value,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
        pst_paths=meta.pst_paths,
        doc_paths=meta.doc_paths,
        total_documents=meta.total_documents,
        total_chunks=meta.total_chunks,
        indexed_at=meta.indexed_at,
        error_message=meta.error_message,
    )


# === 엔드포인트 ===


@router.post("/", response_model=CaseResponse, status_code=201)
async def create_case(request: CaseCreate):
    """새 케이스 생성 + 데이터 소스 경로 설정"""
    store = get_store()

    if not request.name.strip():
        raise HTTPException(status_code=400, detail="케이스 이름은 필수입니다")

    meta = store.create(
        name=request.name.strip(),
        description=request.description.strip(),
        pst_paths=request.pst_paths,
        doc_paths=request.doc_paths,
    )

    logger.info(f"API: 케이스 생성 — {meta.case_id} ({meta.name})")
    return _to_response(meta)


@router.get("/", response_model=list[CaseResponse])
async def list_cases(status: str | None = None):
    """전체 케이스 목록 조회

    Args:
        status: 상태 필터 (created, indexing, ready, archived, error)
    """
    store = get_store()
    cases = store.list_all()

    if status:
        try:
            filter_status = CaseStatus(status)
            cases = [c for c in cases if c.status == filter_status]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"잘못된 상태 값: {status} (허용: created, indexing, ready, archived, error)",
            )

    return [_to_response(c) for c in cases]


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(case_id: str):
    """특정 케이스 상세 조회"""
    store = get_store()
    try:
        meta = store.get(case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"케이스를 찾을 수 없습니다: {case_id}")

    return _to_response(meta)


@router.patch("/{case_id}", response_model=CaseResponse)
async def update_case_sources(case_id: str, request: CaseUpdateSources):
    """데이터 소스 경로 추가"""
    store = get_store()
    try:
        meta = store.update_data_sources(
            case_id,
            pst_paths=request.pst_paths or None,
            doc_paths=request.doc_paths or None,
        )
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"케이스를 찾을 수 없습니다: {case_id}")
    except InvalidStatusTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return _to_response(meta)


@router.delete("/{case_id}", response_model=DeleteResponse)
async def delete_case(case_id: str):
    """케이스 삭제 (벡터DB 컬렉션 + BM25 인덱스 + 메타데이터 삭제)

    인덱싱이 실행 중이면 워커를 먼저 cancel하고 종료될 때까지 대기.
    그렇지 않으면 워커가 삭제 직후 stale 청크를 vectorstore에 다시 upsert하는
    race가 발생할 수 있음.
    """
    store = get_store()

    try:
        store.get(case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"케이스를 찾을 수 없습니다: {case_id}")

    # 인덱싱 실행 중이면 워커부터 정지 (orphan write 방지)
    from src.api.routes.indexing import get_pipeline

    pipeline = get_pipeline()
    if pipeline.is_running(case_id):
        logger.info(f"삭제 전 실행 중 인덱싱 중단: {case_id}")
        finished = pipeline.cancel_and_wait(case_id, timeout=10.0)
        if not finished:
            logger.warning(
                f"인덱싱 워커가 timeout 안에 종료되지 않음: {case_id} — 삭제는 진행"
            )

    # 벡터DB에서 해당 케이스 청크만 삭제 (단일 공유 컬렉션)
    try:
        from src.vectorstore.vector_store import VectorStoreService

        vs = VectorStoreService(case_id=case_id)
        deleted = vs.delete_case_data()
        logger.info(f"벡터DB 케이스 데이터 삭제: case={case_id}, {deleted}개 청크")
    except Exception as e:
        logger.warning(f"벡터DB 케이스 데이터 삭제 실패 (무시): {e}")

    # 케이스 디렉토리 삭제
    store.delete(case_id)

    return DeleteResponse(case_id=case_id, message="케이스가 삭제되었습니다")


@router.post("/{case_id}/archive", response_model=CaseResponse)
async def archive_case(case_id: str):
    """케이스 보관 (읽기 전용 전환)"""
    store = get_store()
    try:
        meta = store.update_status(case_id, CaseStatus.ARCHIVED)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"케이스를 찾을 수 없습니다: {case_id}")
    except InvalidStatusTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return _to_response(meta)
