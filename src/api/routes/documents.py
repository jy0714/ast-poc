"""문서 업로드 및 인덱싱 엔드포인트"""

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

router = APIRouter()


class IndexingStatus(BaseModel):
    total_files: int
    processed: int
    failed: int
    status: str  # "idle" | "processing" | "completed" | "error"


@router.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """문서 파일 업로드 (PST, PDF, DOCX, PPTX, XLSX)"""
    # TODO: 파일 저장 + 파싱 파이프라인 트리거
    return {
        "uploaded": len(files),
        "filenames": [f.filename for f in files],
        "status": "queued",
    }


@router.post("/index")
async def start_indexing():
    """data/input 디렉토리의 파일들을 인덱싱 시작"""
    # TODO: 백그라운드 태스크로 파싱 → 청킹 → 임베딩 → 저장 파이프라인 실행
    return {"status": "indexing_started"}


@router.get("/status", response_model=IndexingStatus)
async def get_indexing_status():
    """현재 인덱싱 진행 상태 조회"""
    # TODO: 실제 상태 조회
    return IndexingStatus(
        total_files=0,
        processed=0,
        failed=0,
        status="idle",
    )
