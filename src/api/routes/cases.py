"""케이스 관리 엔드포인트 (Admin UI)"""

from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class CaseCreate(BaseModel):
    name: str
    description: str = ""
    pst_paths: list[str] = []  # PST 파일 경로들
    doc_paths: list[str] = []  # 문서 폴더 경로들


class CaseResponse(BaseModel):
    case_id: str
    name: str
    description: str
    status: str  # "created" | "indexing" | "ready" | "archived"
    created_at: datetime
    total_documents: int = 0
    total_chunks: int = 0


@router.post("/", response_model=CaseResponse)
async def create_case(request: CaseCreate):
    """새 케이스 생성 + 데이터 소스 경로 설정"""
    # TODO: 케이스 생성, 독립 컬렉션 초기화
    pass


@router.get("/", response_model=list[CaseResponse])
async def list_cases():
    """전체 케이스 목록 조회"""
    # TODO: 케이스 목록 반환
    pass


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(case_id: str):
    """특정 케이스 상세 조회"""
    # TODO: 케이스 상세 정보
    pass


@router.delete("/{case_id}")
async def delete_case(case_id: str):
    """케이스 삭제 (컬렉션 + 인덱스 클렌징)"""
    # TODO: 벡터DB 컬렉션 삭제, BM25 인덱스 삭제
    pass


@router.post("/{case_id}/archive")
async def archive_case(case_id: str):
    """케이스 보관 (읽기 전용 전환)"""
    # TODO: 케이스 상태 변경
    pass
