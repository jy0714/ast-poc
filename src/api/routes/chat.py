"""채팅 (RAG 질의) 엔드포인트 — Analyst UI"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    case_id: str  # 분석 대상 케이스
    message: str
    security_mode: bool = True  # True = 로컬 전용
    filters: dict | None = None  # 메타데이터 필터


class SourceReference(BaseModel):
    content: str  # 참조된 청크 내용
    source_type: str  # "email" | "teams_chat" | "document" | "attachment"
    file_name: str
    date: str | None = None
    participants: list[str] = []
    relevance_score: float = 0.0
    search_method: str = ""  # "vector" | "bm25" | "hybrid"


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    security_mode: bool
    case_id: str


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG 기반 채팅 질의 (특정 케이스 내 검색)"""
    # TODO: 케이스 상태 확인 (ready 상태인지)
    # TODO: 질의 파서 → 하이브리드 검색 → Rank Fusion → LLM
    return ChatResponse(
        answer="[구현 예정] RAG 엔진 연결 후 답변이 생성됩니다.",
        sources=[],
        security_mode=request.security_mode,
        case_id=request.case_id,
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """RAG 기반 스트리밍 채팅 질의"""
    # TODO: SSE 스트리밍 응답
    pass


@router.get("/cases")
async def get_available_cases():
    """분석 가능한 케이스 목록 (status=ready인 것만)"""
    # TODO: 인덱싱 완료된 케이스만 반환
    return []
