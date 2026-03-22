"""인덱싱 관리 엔드포인트 (Admin UI)

- POST /start             — 인덱싱 시작 (백그라운드)
- POST /stop/{case_id}    — 인덱싱 중단
- GET  /progress/{case_id} — 진행률 조회
- POST /increment/{case_id} — 데이터 소스 추가 + 증분 인덱싱
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.cases.case_store import CaseNotFoundError, CaseStore, InvalidStatusTransitionError
from src.indexing.pipeline import IndexingPipeline, IndexingPhase
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 싱글톤 — API 수명 동안 유지하여 진행률 추적
_case_store = CaseStore()
_pipeline = IndexingPipeline(case_store=_case_store)


def get_pipeline() -> IndexingPipeline:
    """인덱싱 파이프라인 인스턴스 반환 (테스트 시 교체 가능)"""
    return _pipeline


def get_case_store() -> CaseStore:
    """케이스 저장소 인스턴스 반환"""
    return _case_store


# === 요청/응답 모델 ===


class IndexingRequest(BaseModel):
    """인덱싱 시작 요청"""

    case_id: str


class IndexingProgress(BaseModel):
    """인덱싱 진행률 응답"""

    case_id: str
    status: str
    phase: str
    total_files: int
    processed_files: int
    total_chunks: int
    progress_percent: float
    elapsed: str = ""
    errors: list[str] = []


class IncrementalRequest(BaseModel):
    """증분 인덱싱 요청"""

    pst_paths: list[str] = []
    doc_paths: list[str] = []


class StopResponse(BaseModel):
    """중단 응답"""

    case_id: str
    message: str


# === 엔드포인트 ===


@router.post("/start", response_model=IndexingProgress)
async def start_indexing(request: IndexingRequest):
    """인덱싱 시작 (백그라운드 스레드)"""
    pipeline = get_pipeline()
    store = get_case_store()

    # 케이스 존재 확인
    try:
        case_meta = store.get(request.case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"케이스를 찾을 수 없습니다: {request.case_id}")

    # 이미 실행 중인지 확인
    if pipeline.is_running(request.case_id):
        raise HTTPException(status_code=409, detail="이미 인덱싱이 진행 중입니다")

    # 보관 상태 체크
    from src.cases.case_store import CaseStatus

    if case_meta.status == CaseStatus.ARCHIVED:
        raise HTTPException(status_code=409, detail="보관된 케이스는 인덱싱할 수 없습니다")

    # 백그라운드 실행
    progress = pipeline.run_async(request.case_id)
    progress_dict = progress.to_dict()

    logger.info(f"API: 인덱싱 시작 — {request.case_id}")
    return IndexingProgress(**progress_dict)


@router.post("/stop/{case_id}", response_model=StopResponse)
async def stop_indexing(case_id: str):
    """인덱싱 중단"""
    pipeline = get_pipeline()

    if not pipeline.is_running(case_id):
        raise HTTPException(status_code=409, detail="실행 중인 인덱싱이 없습니다")

    success = pipeline.cancel(case_id)
    if not success:
        raise HTTPException(status_code=500, detail="인덱싱 중단에 실패했습니다")

    return StopResponse(case_id=case_id, message="인덱싱 중단 요청이 전달되었습니다")


@router.get("/progress/{case_id}", response_model=IndexingProgress)
async def get_indexing_progress(case_id: str):
    """인덱싱 진행률 조회"""
    pipeline = get_pipeline()
    store = get_case_store()

    # 케이스 존재 확인
    try:
        store.get(case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"케이스를 찾을 수 없습니다: {case_id}")

    progress = pipeline.get_progress(case_id)
    return IndexingProgress(**progress.to_dict())


@router.post("/increment/{case_id}", response_model=IndexingProgress)
async def add_incremental_data(case_id: str, request: IncrementalRequest):
    """추가 데이터 소스 등록 + 증분 인덱싱 트리거"""
    pipeline = get_pipeline()
    store = get_case_store()

    # 케이스 존재 확인
    try:
        store.get(case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"케이스를 찾을 수 없습니다: {case_id}")

    # 이미 실행 중인지 확인
    if pipeline.is_running(case_id):
        raise HTTPException(status_code=409, detail="이미 인덱싱이 진행 중입니다")

    # 데이터 소스 추가
    if not request.pst_paths and not request.doc_paths:
        raise HTTPException(status_code=400, detail="추가할 데이터 소스가 없습니다")

    try:
        store.update_data_sources(
            case_id,
            pst_paths=request.pst_paths or None,
            doc_paths=request.doc_paths or None,
        )
    except InvalidStatusTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # 인덱싱 시작
    progress = pipeline.run_async(case_id)
    progress_dict = progress.to_dict()

    logger.info(f"API: 증분 인덱싱 시작 — {case_id}")
    return IndexingProgress(**progress_dict)
