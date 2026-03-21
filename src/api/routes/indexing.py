"""인덱싱 관리 엔드포인트 (Admin UI)"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class IndexingRequest(BaseModel):
    case_id: str
    incremental: bool = False  # True면 증분 인덱싱


class IndexingProgress(BaseModel):
    case_id: str
    status: str  # "idle" | "parsing" | "chunking" | "embedding" | "storing" | "completed" | "error"
    phase: str  # 현재 단계
    total_files: int
    processed_files: int
    total_chunks: int
    progress_percent: float
    estimated_remaining: str  # "약 2시간 30분"
    errors: list[str] = []


@router.post("/start")
async def start_indexing(request: IndexingRequest):
    """인덱싱 시작 (전체 또는 증분)"""
    # TODO: 백그라운드 태스크로 인덱싱 파이프라인 실행
    pass


@router.post("/stop/{case_id}")
async def stop_indexing(case_id: str):
    """인덱싱 중단"""
    # TODO: 실행 중인 인덱싱 중단
    pass


@router.get("/progress/{case_id}", response_model=IndexingProgress)
async def get_indexing_progress(case_id: str):
    """인덱싱 진행률 조회"""
    # TODO: 실시간 진행 상태 반환
    pass


@router.post("/increment/{case_id}")
async def add_incremental_data(case_id: str, paths: list[str]):
    """추가 데이터 소스 지정 + 증분 인덱싱 트리거"""
    # TODO: 추가 파일 경로 등록 + 증분 인덱싱
    pass
