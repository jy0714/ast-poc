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

import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

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
    # 이메일 전용 — 비-이메일 출처에서는 빈 값
    sender: str = ""
    recipients: list[str] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)
    message_id: str = ""
    in_reply_to: str = ""
    # Office/PDF 작성자·수정자 추적용
    author: str = ""
    last_modified_by: str = ""
    created_date: str = ""
    last_modified: str = ""


@dataclass
class QueryResult:
    """RAG 질의 결과"""

    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    case_id: str = ""
    secure_mode: bool = True
    # 출처 인용 검증 (할루시네이션 감지)
    citation_count: int = 0  # 응답에서 발견된 [출처 N] 개수
    invalid_citations: list[int] = field(default_factory=list)  # 존재하지 않는 출처 번호
    uncited_response: bool = False  # 인용이 전혀 없는 (거절이 아닌) 응답


# [출처 N] / [출처N] / [출처 1][출처 3] 모두 매칭
_CITATION_RE = re.compile(r"\[출처\s*(\d+)\]")
# 정당하게 인용이 없을 수 있는 거절 답변 패턴
_REFUSAL_PATTERNS = (
    "찾을 수 없습니다",
    "검색 결과가 없습니다",
    "자료에서 해당",
    "확인할 수 없습니다",
)


def validate_citations(answer: str, num_sources: int) -> tuple[int, list[int], bool]:
    """LLM 응답의 출처 인용을 검증

    Args:
        answer: LLM 응답 텍스트
        num_sources: 실제 제공된 출처 수

    Returns:
        (citation_count, invalid_citations, uncited_response)
        - citation_count: 발견된 [출처 N] 총 개수
        - invalid_citations: 1..num_sources 범위를 벗어난 출처 번호 (정렬, 중복 제거)
        - uncited_response: 인용이 0개이고 거절 답변도 아닌 경우 (할루시네이션 위험)
    """
    matches = _CITATION_RE.findall(answer or "")
    citation_count = len(matches)
    nums = [int(m) for m in matches]
    invalid = sorted({n for n in nums if n < 1 or n > num_sources})
    is_refusal = any(p in (answer or "") for p in _REFUSAL_PATTERNS)
    uncited = citation_count == 0 and not is_refusal
    return citation_count, invalid, uncited


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
        self.vector_store = vector_store or VectorStoreService(case_id=case_id)
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
            # reranker 미사용 시 RRF 점수 상대 필터 — 상위 결과 평균 대비 너무 낮은
            # 결과 제거 (절대 임계값 rrf_min_score는 vector_store에서 이미 적용됨)
            results = self._apply_relative_score_filter(results)

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
                    sender=meta.get("sender", ""),
                    recipients=meta.get("recipients", []),
                    cc=meta.get("cc", []),
                    attachments=meta.get("attachment_filenames", []),
                    message_id=meta.get("message_id", ""),
                    in_reply_to=meta.get("in_reply_to", ""),
                    author=meta.get("author", ""),
                    last_modified_by=meta.get("last_modified_by", ""),
                    created_date=meta.get("created_date", ""),
                    last_modified=meta.get("last_modified", ""),
                )
            )

        return sources

    @staticmethod
    def _apply_relative_score_filter(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """상위 결과 평균 점수의 30% 미만인 결과 제거 (상대 필터)

        reranker 미사용 시 RRF 점수 기반으로 적용. 결과가 1건 이하면 필터하지 않음
        (비교 기준이 자기 자신뿐).
        """
        if len(results) <= 1:
            return results
        scores = [r.get("score", 0.0) for r in results]
        mean_score = sum(scores) / len(scores)
        if mean_score <= 0:
            return results
        threshold = mean_score * 0.3
        filtered = [r for r in results if r.get("score", 0.0) >= threshold]
        if filtered and len(filtered) < len(results):
            logger.info(
                f"RRF 상대 필터: {len(results)}건 중 {len(filtered)}건 통과 "
                f"(평균 {mean_score:.4f}의 30%={threshold:.4f} 미만 제거)"
            )
        # 필터가 전부 제거하는 경우는 방어적으로 원본 유지 (상대 필터가 과하게 작동)
        return filtered or results

    def _postprocess_sources(
        self, search_query: str, sources: list[SourceReference]
    ) -> tuple[list[SourceReference], str | None]:
        """LLM 전달 전 소스 후처리 (query/query_stream 공용)

        1. 짧은 청크 제거 (#3) — content가 min_source_length 미만이면 과도한
           맥락 생성 위험이 있어 제거.
        2. 관련성 사전 체크 (#5) — 질의 키워드가 검색 결과에 하나도 없으면
           프롬프트에 삽입할 경고 문자열 반환.

        Returns:
            (필터링된 sources, extra_system_warning | None)
        """
        # #3 짧은 청크 필터
        min_len = settings.min_source_length
        filtered = [s for s in sources if len(s.content) >= min_len]
        removed = len(sources) - len(filtered)
        if removed:
            logger.info(
                f"짧은 청크 필터링: {removed}건 제거 "
                f"(min_source_length={min_len}, {len(filtered)}건 잔존)"
            )

        # #5 관련성 체크
        extra_warning: str | None = None
        if settings.enable_relevance_check and filtered:
            if not self._has_keyword_overlap(search_query, filtered):
                logger.warning(
                    f"검색 결과와 질문의 키워드 일치도가 낮습니다 (case={self.case_id})"
                )
                extra_warning = (
                    "주의: 검색 결과가 질문과 직접적으로 관련되지 않을 수 있습니다. "
                    "확실한 근거가 없으면 반드시 '찾을 수 없습니다'로 답하세요."
                )

        return filtered, extra_warning

    @staticmethod
    def _has_keyword_overlap(query: str, sources: list[SourceReference]) -> bool:
        """질의 핵심 키워드가 검색 결과 content에 하나라도 포함되는지

        키워드를 추출하지 못하면(너무 짧은 질의 등) True 반환 (체크 skip).
        """
        from src.chunkers.metadata_enricher import extract_topics

        keywords = extract_topics(query, max_topics=5)
        if not keywords:
            return True
        combined = " ".join(s.content for s in sources).lower()
        return any(k.lower() in combined for k in keywords)

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

        # 2. 후처리 (짧은 청크 제거 + 관련성 체크)
        sources, extra_warning = self._postprocess_sources(search_query, sources)

        if not sources:
            return QueryResult(
                answer="검색 결과가 없습니다. 질문을 다시 작성하거나 다른 케이스를 선택해 주세요.",
                sources=[],
                case_id=self.case_id,
                secure_mode=is_secure,
            )

        # 3. 컨텍스트 조합
        context_texts = [self._format_source_context(s) for s in sources]

        # 검색 결과 (LLM에 전달되는 context) 로깅
        self._log_sources(question, sources, parsed.intent)

        # 4. LLM 응답 생성 (관련성 경고가 있으면 프롬프트에 삽입)
        try:
            answer = await self.llm_router.generate(
                question=question,  # LLM에는 원본 질의 전달
                context=context_texts,
                secure_mode=is_secure,
                extra_system_warning=extra_warning,
            )
        except Exception as e:
            logger.error(f"LLM 응답 생성 실패: {e}")
            answer = (
                f"LLM 응답 생성에 실패했습니다: {e}\n\n"
                "아래 검색 결과를 직접 확인해 주세요."
            )

        # 5. 출처 인용 검증 (할루시네이션 감지)
        citation_count, invalid_citations, uncited_response = validate_citations(
            answer, len(sources)
        )
        if invalid_citations:
            logger.warning(
                f"가짜 인용 감지: [출처 {invalid_citations}]가 실제 출처 수 "
                f"{len(sources)}개를 초과 (case={self.case_id})"
            )
        if uncited_response:
            logger.warning(
                f"LLM이 출처 인용 없이 답변 — 할루시네이션 위험 (case={self.case_id})"
            )

        # 최종 답변 로깅 (context와 비교 가능하도록)
        logger.info(
            f"RAG 질의 완료: case={self.case_id}, "
            f"검색={len(sources)}건, 보안={is_secure}, 의도={parsed.intent}, "
            f"인용={citation_count}건"
        )
        logger.debug(f"[답변] case={self.case_id}\n{answer}")

        return QueryResult(
            answer=answer,
            sources=sources,
            case_id=self.case_id,
            secure_mode=is_secure,
            citation_count=citation_count,
            invalid_citations=invalid_citations,
            uncited_response=uncited_response,
        )

    async def query_stream(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        secure_mode: bool | None = None,
        n_results: int | None = None,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> tuple[AsyncIterator[str], list[SourceReference]]:
        """스트리밍 RAG 질의: 검색 → LLM 스트리밍 응답

        Args:
            is_disconnected: 클라이언트 연결 끊김 여부를 반환하는 async 콜백 (선택).
                검색 시작 전과 LLM 스트림 시작 직전(=reranker 완료 후)에 체크하여,
                이미 끊겼으면 LLM을 호출하지 않고 빈 스트림을 반환한다. None이면 기존 동작.

        Returns:
            (토큰 스트림, 출처 리스트) 튜플. 중단 시 빈 토큰 스트림.
        """
        is_secure = secure_mode if secure_mode is not None else settings.is_secure_mode

        async def _empty_stream() -> AsyncIterator[str]:
            return
            yield  # noqa — async generator로 만들기 위한 unreachable yield

        # 검색 시작 전 체크 — 사용자가 전송 직후 바로 멈춘 경우
        if is_disconnected is not None and await is_disconnected():
            logger.info(f"query_stream 중단 (검색 전): case={self.case_id}")
            return _empty_stream(), []

        # 0. 질의 파싱
        parsed = parse_query(question)
        search_query = parsed.cleaned
        merged_filters = _merge_filters(parsed.filters, filters)

        # 1. 하이브리드 검색 (+ reranker가 활성화면 search 내부에서 수행)
        sources = self.search(query=search_query, filters=merged_filters, n_results=n_results)

        # 2. 후처리 (짧은 청크 제거 + 관련성 체크) — query와 동일 정책
        sources, extra_warning = self._postprocess_sources(search_query, sources)

        # 검색 결과 (LLM에 전달되는 context) 로깅
        self._log_sources(question, sources, parsed.intent)

        # LLM 스트림 시작 직전 체크 — 검색/reranker가 느려 그 사이 사용자가 멈춘 경우.
        # 비싼 LLM 호출 전에 끊어 GPU 자원을 아낀다.
        if is_disconnected is not None and await is_disconnected():
            logger.info(f"query_stream 중단 (LLM 전): case={self.case_id}")
            return _empty_stream(), sources

        # 3. 컨텍스트 조합
        context_texts = [self._format_source_context(s) for s in sources]

        # 4. LLM 스트리밍 (관련성 경고가 있으면 프롬프트에 삽입)
        token_stream = self.llm_router.generate_stream(
            question=question,  # LLM에는 원본 질의 전달
            context=context_texts,
            secure_mode=is_secure,
            extra_system_warning=extra_warning,
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
        """출처 정보를 LLM 컨텍스트 텍스트로 포맷

        이메일은 From/To/Cc/첨부를 별도 노출 (LLM이 발신자/수신자 관계를
        정확히 식별할 수 있도록). 그 외 소스는 기존 방식 유지.
        """
        header_parts: list[str] = []

        type_labels = {
            "email": "이메일",
            "teams_chat": "Teams 채팅",
            "document": "문서",
            "attachment": "첨부파일",
        }
        if source.source_type:
            header_parts.append(
                f"유형: {type_labels.get(source.source_type, source.source_type)}"
            )
        if source.filename:
            header_parts.append(f"파일: {source.filename}")
        if source.subject:
            header_parts.append(f"제목: {source.subject}")
        if source.date:
            header_parts.append(f"날짜: {source.date}")

        if source.source_type == "email":
            if source.sender:
                header_parts.append(f"From: {source.sender}")
            if source.recipients:
                header_parts.append(f"To: {', '.join(source.recipients[:5])}")
            if source.cc:
                header_parts.append(f"Cc: {', '.join(source.cc[:5])}")
            if source.attachments:
                header_parts.append(f"첨부: {', '.join(source.attachments[:5])}")
        else:
            # 문서/첨부: 작성자/수정자 추적 정보
            if source.author:
                header_parts.append(f"작성자: {source.author}")
            if source.last_modified_by and source.last_modified_by != source.author:
                header_parts.append(f"마지막 수정: {source.last_modified_by}")
            if source.created_date:
                header_parts.append(f"생성일: {source.created_date}")
            if source.last_modified and source.last_modified != source.created_date:
                header_parts.append(f"수정일: {source.last_modified}")
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
