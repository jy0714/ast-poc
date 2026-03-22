"""채팅 (RAG 질의) 엔드포인트 — Analyst UI

- POST /           — RAG 질의 (동기 응답)
- POST /stream      — RAG 질의 (SSE 스트리밍)
- GET  /cases       — 분석 가능한 케이스 목록 (status=ready)
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.cases.case_store import CaseNotFoundError, CaseStatus, CaseStore
from src.rag.engine import RAGEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

_case_store = CaseStore()


def get_case_store() -> CaseStore:
    return _case_store


# === 요청/응답 모델 ===


class ChatRequest(BaseModel):
    """RAG 질의 요청"""

    case_id: str
    message: str
    security_mode: bool = True
    filters: dict | None = None


class SourceReference(BaseModel):
    """검색 결과 출처"""

    content: str
    source_type: str
    filename: str = ""
    date: str = ""
    participants: list[str] = []
    subject: str = ""
    relevance_score: float = 0.0
    search_method: str = ""


class ChatResponse(BaseModel):
    """RAG 응답"""

    answer: str
    sources: list[SourceReference]
    security_mode: bool
    case_id: str


class CaseInfo(BaseModel):
    """분석 가능 케이스 정보"""

    case_id: str
    name: str
    description: str
    total_documents: int
    total_chunks: int


# === 엔드포인트 ===


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG 기반 채팅 질의 (특정 케이스 내 검색)"""
    store = get_case_store()

    # 케이스 존재 + ready 상태 확인
    try:
        case_meta = store.get(request.case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"케이스를 찾을 수 없습니다: {request.case_id}")

    if case_meta.status not in (CaseStatus.READY, CaseStatus.ARCHIVED):
        raise HTTPException(
            status_code=409,
            detail=f"이 케이스는 아직 질의할 수 없습니다 (상태: {case_meta.status.value}). "
            "인덱싱이 완료된 케이스만 질의 가능합니다.",
        )

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해 주세요")

    # RAG 질의
    engine = RAGEngine(case_id=request.case_id)
    result = await engine.query(
        question=request.message.strip(),
        filters=request.filters,
        secure_mode=request.security_mode,
    )

    sources = [
        SourceReference(
            content=s.content[:500],  # 응답 크기 제한
            source_type=s.source_type,
            filename=s.filename,
            date=s.date,
            participants=s.participants[:10],
            subject=s.subject,
            relevance_score=round(s.score, 4),
            search_method=s.search_method,
        )
        for s in result.sources
    ]

    return ChatResponse(
        answer=result.answer,
        sources=sources,
        security_mode=result.secure_mode,
        case_id=result.case_id,
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """RAG 기반 스트리밍 채팅 질의 (SSE)"""
    store = get_case_store()

    try:
        case_meta = store.get(request.case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"케이스를 찾을 수 없습니다: {request.case_id}")

    if case_meta.status not in (CaseStatus.READY, CaseStatus.ARCHIVED):
        raise HTTPException(
            status_code=409,
            detail=f"이 케이스는 아직 질의할 수 없습니다 (상태: {case_meta.status.value})",
        )

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해 주세요")

    engine = RAGEngine(case_id=request.case_id)

    async def event_generator():
        try:
            token_stream, sources = await engine.query_stream(
                question=request.message.strip(),
                filters=request.filters,
                secure_mode=request.security_mode,
            )

            # 토큰 스트리밍
            async for token in token_stream:
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

            # 출처 정보 전송
            sources_data = [
                {
                    "content": s.content[:500],
                    "source_type": s.source_type,
                    "filename": s.filename,
                    "date": s.date,
                    "participants": s.participants[:10],
                    "subject": s.subject,
                    "relevance_score": round(s.score, 4),
                    "search_method": s.search_method,
                }
                for s in sources
            ]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"스트리밍 에러: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/cases", response_model=list[CaseInfo])
async def get_available_cases():
    """분석 가능한 케이스 목록 (인덱싱 완료된 것만)"""
    store = get_case_store()
    all_cases = store.list_all()

    available = [
        CaseInfo(
            case_id=c.case_id,
            name=c.name,
            description=c.description,
            total_documents=c.total_documents,
            total_chunks=c.total_chunks,
        )
        for c in all_cases
        if c.status in (CaseStatus.READY, CaseStatus.ARCHIVED)
    ]

    return available
