"""RAG 질의 엔진 — 하이브리드 검색 + LLM 응답 생성

흐름:
    사용자 질의
    → 질의 파서 (의도 분석 + 필터 추출)
    → 하이브리드 검색 (벡터 + BM25 + RRF)
    → 컨텍스트 조합
    → LLM 응답 생성
    → 출처 포함 결과 반환
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from src.llm.router import LLMRouter
from src.rag.query_parser import ParsedQuery, parse_query
from src.vectorstore.vector_store import VectorStoreService
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SourceReference:
    """검색 결과 출처 정보"""

    content: str
    source_type: str  # "email" | "teams_chat" | "document" | "attachment"
    filename: str = ""
    date: str = ""
    participants: list[str] = field(default_factory=list)
    subject: str = ""
    score: float = 0.0
    search_method: str = ""  # "vector" | "bm25" | "hybrid"


@dataclass
class QueryResult:
    """RAG 질의 결과"""

    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    case_id: str = ""
    secure_mode: bool = True


class RAGEngine:
    """RAG 질의 엔진 — 검색 + LLM 응답 생성

    사용법:
        engine = RAGEngine(case_id="abc123")
        result = await engine.query("감사 보고서에서 비용 관련 내용은?")
        print(result.answer)
        for src in result.sources:
            print(f"  [{src.source_type}] {src.filename}: {src.content[:100]}")
    """

    def __init__(
        self,
        case_id: str,
        vector_store: VectorStoreService | None = None,
        llm_router: LLMRouter | None = None,
        top_k: int | None = None,
        rerank_enabled: bool | None = None,
    ) -> None:
        """RAGEngine 초기화

        Args:
            case_id: 검색 대상 케이스 ID
            vector_store: 벡터 저장소 (기본: case_id 기반 생성)
            llm_router: LLM 라우터 (기본: 새 인스턴스)
            top_k: 검색 결과 수 (기본: settings.search_top_k)
            rerank_enabled: Reranker 사용 여부 (None이면 settings.rerank_enabled)
        """
        self.case_id = case_id
        self.vector_store = vector_store or VectorStoreService(
            collection_name=f"case_{case_id}"
        )
        self.llm_router = llm_router or LLMRouter()
        self.top_k = top_k or settings.search_top_k
        self.rerank_enabled = (
            rerank_enabled if rerank_enabled is not None else settings.rerank_enabled
        )

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        n_results: int | None = None,
    ) -> list[SourceReference]:
        """하이브리드 검색만 수행 (LLM 호출 없이)

        rerank_enabled=True일 경우:
            1. rerank_top_k_candidates개 후보를 하이브리드 검색으로 가져옴
            2. Reranker로 rerank_top_n개 선별
        rerank_enabled=False일 경우:
            기존 동작 (하이브리드 검색 → top_k 반환)

        Args:
            query: 검색 쿼리
            filters: 메타데이터 필터 (source_type, date 등)
            n_results: 결과 수

        Returns:
            SourceReference 리스트 (점수 내림차순)
        """
        if self.rerank_enabled:
            # reranker용: 넉넉한 후보를 가져온 뒤 reranker가 선별
            candidate_count = settings.rerank_top_k_candidates
            results = self.vector_store.search(
                query=query,
                n_results=candidate_count,
                filters=filters,
            )

            if results:
                from src.rag.reranker import get_reranker

                reranker = get_reranker()
                final_n = n_results or settings.rerank_top_n
                results = reranker.rerank(
                    query=query, documents=results, top_n=final_n
                )
        else:
            results = self.vector_store.search(
                query=query,
                n_results=n_results or self.top_k,
                filters=filters,
            )

        sources: list[SourceReference] = []
        for r in results:
            meta = r.get("metadata", {})
            sources.append(
                SourceReference(
                    content=r.get("content", ""),
                    source_type=meta.get("source_type", "unknown"),
                    filename=meta.get("filename", ""),
                    date=meta.get("date", meta.get("date_range_start", "")),
                    participants=meta.get("participants", []),
                    subject=meta.get("subject", meta.get("thread_subject", "")),
                    score=r.get("rerank_score", r.get("score", 0.0)),
                    search_method=r.get("search_method", ""),
                )
            )

        return sources

    async def query(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        secure_mode: bool | None = None,
        n_results: int | None = None,
    ) -> QueryResult:
        """RAG 질의: 검색 → 컨텍스트 조합 → LLM 응답

        Args:
            question: 사용자 질문
            filters: 메타데이터 필터
            secure_mode: 보안 모드 (None이면 설정값 사용)
            n_results: 검색 결과 수

        Returns:
            QueryResult (답변 + 출처)
        """
        is_secure = secure_mode if secure_mode is not None else settings.is_secure_mode

        # 0. 질의 파싱 (필터 추출 + 쿼리 정제)
        parsed = parse_query(question)
        search_query = parsed.cleaned
        merged_filters = _merge_filters(parsed.filters, filters)

        logger.debug(
            f"질의 파싱: intent={parsed.intent}, "
            f"parsed_filters={parsed.filters}, merged_filters={merged_filters}"
        )

        # 1. 하이브리드 검색
        sources = self.search(query=search_query, filters=merged_filters, n_results=n_results)

        if not sources:
            return QueryResult(
                answer="검색 결과가 없습니다. 질문을 다시 작성하거나 다른 케이스를 선택해 주세요.",
                sources=[],
                case_id=self.case_id,
                secure_mode=is_secure,
            )

        # 2. 컨텍스트 조합
        context_texts = [self._format_source_context(s) for s in sources]

        # 검색 결과 (LLM에 전달되는 context) 로깅
        self._log_sources(question, sources, parsed.intent)

        # 3. LLM 응답 생성
        try:
            answer = await self.llm_router.generate(
                question=question,  # LLM에는 원본 질의 전달
                context=context_texts,
                secure_mode=is_secure,
            )
        except Exception as e:
            logger.error(f"LLM 응답 생성 실패: {e}")
            answer = (
                f"LLM 응답 생성에 실패했습니다: {e}\n\n"
                "아래 검색 결과를 직접 확인해 주세요."
            )

        # 최종 답변 로깅 (context와 비교 가능하도록)
        logger.info(
            f"RAG 질의 완료: case={self.case_id}, "
            f"검색={len(sources)}건, 보안={is_secure}, 의도={parsed.intent}"
        )
        logger.debug(f"[답변] case={self.case_id}\n{answer}")

        return QueryResult(
            answer=answer,
            sources=sources,
            case_id=self.case_id,
            secure_mode=is_secure,
        )

    async def query_stream(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        secure_mode: bool | None = None,
        n_results: int | None = None,
    ) -> tuple[AsyncIterator[str], list[SourceReference]]:
        """스트리밍 RAG 질의: 검색 → LLM 스트리밍 응답

        Returns:
            (토큰 스트림, 출처 리스트) 튜플
        """
        is_secure = secure_mode if secure_mode is not None else settings.is_secure_mode

        # 0. 질의 파싱
        parsed = parse_query(question)
        search_query = parsed.cleaned
        merged_filters = _merge_filters(parsed.filters, filters)

        # 1. 하이브리드 검색
        sources = self.search(query=search_query, filters=merged_filters, n_results=n_results)

        # 검색 결과 (LLM에 전달되는 context) 로깅
        self._log_sources(question, sources, parsed.intent)

        # 2. 컨텍스트 조합
        context_texts = [self._format_source_context(s) for s in sources]

        # 3. LLM 스트리밍
        token_stream = self.llm_router.generate_stream(
            question=question,  # LLM에는 원본 질의 전달
            context=context_texts,
            secure_mode=is_secure,
        )

        return token_stream, sources

    def _log_sources(
        self,
        question: str,
        sources: list[SourceReference],
        intent: str = "",
    ) -> None:
        """검색된 청크(sources)를 로깅 — LLM에 전달되는 context 추적용"""
        logger.info(
            f"[검색 context] case={self.case_id}, "
            f"질의='{question[:80]}', 의도={intent}, 출처={len(sources)}건"
        )
        for i, s in enumerate(sources):
            logger.debug(
                f"  [출처 {i + 1}] type={s.source_type}, file={s.filename}, "
                f"score={s.score:.4f}, method={s.search_method}, "
                f"subject={s.subject or '-'}, date={s.date or '-'}\n"
                f"    content: {s.content[:200]}{'...' if len(s.content) > 200 else ''}"
            )

    @staticmethod
    def _format_source_context(source: SourceReference) -> str:
        """출처 정보를 LLM 컨텍스트 텍스트로 포맷"""
        header_parts: list[str] = []

        if source.source_type:
            type_labels = {
                "email": "이메일",
                "teams_chat": "Teams 채팅",
                "document": "문서",
                "attachment": "첨부파일",
            }
            header_parts.append(
                f"유형: {type_labels.get(source.source_type, source.source_type)}"
            )
        if source.filename:
            header_parts.append(f"파일: {source.filename}")
        if source.subject:
            header_parts.append(f"제목: {source.subject}")
        if source.date:
            header_parts.append(f"날짜: {source.date}")
        if source.participants:
            header_parts.append(f"참여자: {', '.join(source.participants[:5])}")

        header = " | ".join(header_parts) if header_parts else "출처 불명"
        return f"[{header}]\n{source.content}"


def _merge_filters(
    parsed: dict[str, Any],
    explicit: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """파서 추출 필터와 API 명시 필터를 병합 (명시 필터 우선)"""
    if not parsed and not explicit:
        return None
    merged = dict(parsed) if parsed else {}
    if explicit:
        merged.update(explicit)  # 명시 필터가 파서 필터를 덮어씀
    return merged or None
