"""채팅 (RAG 질의) 엔드포인트 — Analyst UI

- POST /           — RAG 질의 (동기 응답)
- POST /stream      — RAG 질의 (SSE 스트리밍)
- GET  /cases       — 분석 가능한 케이스 목록 (status=ready)
- GET  /history/{case_id} — 채팅 히스토리 조회
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
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
    rerank: bool | None = None  # None이면 전역 설정(settings.rerank_enabled) 사용


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
    # 이메일 전용 필드 (다른 source_type은 빈 값)
    sender: str = ""
    recipients: list[str] = []
    cc: list[str] = []
    attachments: list[str] = []
    message_id: str = ""
    in_reply_to: str = ""
    # Office/PDF 작성자·수정자 추적
    author: str = ""
    last_modified_by: str = ""
    created_date: str = ""
    last_modified: str = ""


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


class ChatHistoryItem(BaseModel):
    """채팅 히스토리 항목"""

    id: int
    question: str
    answer: str
    security_mode: bool
    created_at: str
    sources_count: int
    is_stopped: bool = False


# === 채팅 히스토리 저장 ===


def _save_chat_history(
    case_id: str,
    question: str,
    answer: str,
    security_mode: bool,
    filters: dict | None,
    sources: list[SourceReference],
    is_stopped: bool = False,
) -> None:
    """채팅 히스토리를 DB에 저장

    Args:
        is_stopped: 사용자가 중단한 응답이면 True (지금까지 모은 answer를 그대로 저장).
    """
    try:
        from src.db.database import get_session
        from src.db.models import ChatHistoryModel, ChatSourceModel

        chat_record = ChatHistoryModel(
            case_id=case_id,
            question=question,
            answer=answer,
            security_mode=1 if security_mode else 0,
            filters=json.dumps(filters or {}, ensure_ascii=False),
            is_stopped=1 if is_stopped else 0,
        )

        with get_session() as session:
            session.add(chat_record)
            session.flush()  # chat_record.id 할당

            for src in sources:
                source_record = ChatSourceModel(
                    chat_id=chat_record.id,
                    content=src.content[:500],
                    source_type=src.source_type,
                    filename=src.filename,
                    date=src.date,
                    participants=json.dumps(src.participants[:10], ensure_ascii=False),
                    subject=src.subject,
                    relevance_score=src.relevance_score,
                    search_method=src.search_method,
                    sender=src.sender,
                    recipients=json.dumps(src.recipients[:10], ensure_ascii=False),
                    cc=json.dumps(src.cc[:10], ensure_ascii=False),
                    attachments=json.dumps(src.attachments[:10], ensure_ascii=False),
                    message_id=src.message_id,
                    in_reply_to=src.in_reply_to,
                    author=src.author,
                    last_modified_by=src.last_modified_by,
                    created_date=src.created_date,
                    last_modified=src.last_modified,
                )
                session.add(source_record)

            session.commit()

    except Exception as e:
        # 응답 품질에는 영향 없으나 데이터 유실이므로 ERROR로 기록
        # (운영 단계에서 메트릭 카운터로 알람 연동 예정)
        logger.error(
            f"채팅 히스토리 저장 실패 — case={case_id}, q='{question[:80]}': {e}",
            exc_info=True,
        )


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
    engine = RAGEngine(
        case_id=request.case_id,
        rerank_enabled=request.rerank,
    )
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
            sender=s.sender,
            recipients=s.recipients[:10],
            cc=s.cc[:10],
            attachments=s.attachments[:10],
            message_id=s.message_id,
            in_reply_to=s.in_reply_to,
            author=s.author,
            last_modified_by=s.last_modified_by,
            created_date=s.created_date,
            last_modified=s.last_modified,
        )
        for s in result.sources
    ]

    # 채팅 히스토리 저장
    _save_chat_history(
        case_id=request.case_id,
        question=request.message.strip(),
        answer=result.answer,
        security_mode=request.security_mode,
        filters=request.filters,
        sources=sources,
    )

    return ChatResponse(
        answer=result.answer,
        sources=sources,
        security_mode=result.secure_mode,
        case_id=result.case_id,
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest, raw_request: Request):
    """RAG 기반 스트리밍 채팅 질의 (SSE)

    클라이언트가 연결을 끊으면(AbortController.abort 또는 네트워크 단절)
    raw_request.is_disconnected()가 True가 되어 토큰 생성을 멈추고 리소스를 해제한다.
    중단 시점까지 모은 답변은 is_stopped=True로 히스토리에 저장된다.
    """
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

    engine = RAGEngine(
        case_id=request.case_id,
        rerank_enabled=request.rerank,
    )

    async def event_generator():
        collected_answer = ""
        collected_sources: list[SourceReference] = []
        stopped = False

        async def check_disconnected() -> bool:
            return await raw_request.is_disconnected()

        try:
            # 검색/reranker 단계의 조기 중단을 위해 engine에 콜백 전달
            token_stream, sources = await engine.query_stream(
                question=request.message.strip(),
                filters=request.filters,
                secure_mode=request.security_mode,
                is_disconnected=check_disconnected,
            )

            # 토큰 스트리밍 — 매 토큰마다 연결 상태 확인
            async for token in token_stream:
                if await raw_request.is_disconnected():
                    stopped = True
                    logger.info(
                        f"스트리밍 중단 감지 (client disconnect): case={request.case_id}, "
                        f"지금까지 {len(collected_answer)}자"
                    )
                    break
                collected_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

            if not stopped:
                # 출처 정보 전송 (정상 완료 시에만)
                sources_data = []
                for s in sources:
                    src_ref = SourceReference(
                        content=s.content[:500],
                        source_type=s.source_type,
                        filename=s.filename,
                        date=s.date,
                        participants=s.participants[:10],
                        subject=s.subject,
                        relevance_score=round(s.score, 4),
                        search_method=s.search_method,
                        sender=s.sender,
                        recipients=s.recipients[:10],
                        cc=s.cc[:10],
                        attachments=s.attachments[:10],
                        message_id=s.message_id,
                        in_reply_to=s.in_reply_to,
                        author=s.author,
                        last_modified_by=s.last_modified_by,
                        created_date=s.created_date,
                        last_modified=s.last_modified,
                    )
                    collected_sources.append(src_ref)
                    sources_data.append(src_ref.model_dump())

                yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            # 정상 완료/중단 모두 히스토리 저장 (중단도 사용자에게 의미있는 부분 응답)
            _save_chat_history(
                case_id=request.case_id,
                question=request.message.strip(),
                answer=collected_answer,
                security_mode=request.security_mode,
                filters=request.filters,
                sources=collected_sources,
                is_stopped=stopped,
            )

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


@router.get("/history/{case_id}", response_model=list[ChatHistoryItem])
async def get_chat_history(case_id: str, limit: int = 50):
    """케이스별 채팅 히스토리 조회"""
    store = get_case_store()

    try:
        store.get(case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"케이스를 찾을 수 없습니다: {case_id}")

    try:
        from src.db.database import get_session
        from src.db.models import ChatHistoryModel

        with get_session() as session:
            rows = (
                session.query(ChatHistoryModel)
                .filter(ChatHistoryModel.case_id == case_id)
                .order_by(ChatHistoryModel.created_at.desc())
                .limit(limit)
                .all()
            )

            return [
                ChatHistoryItem(
                    id=r.id,
                    question=r.question,
                    answer=r.answer,
                    security_mode=bool(r.security_mode),
                    created_at=r.created_at.isoformat(),
                    sources_count=len(r.sources),
                    is_stopped=bool(getattr(r, "is_stopped", 0)),
                )
                for r in rows
            ]
    except Exception as e:
        logger.warning(f"채팅 히스토리 조회 실패: {e}")
        return []
