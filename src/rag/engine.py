"""RAG 질의 엔진 — 검색(retrieve) + LLM 응답 생성

흐름:
    사용자 질의
    → 질의 파서 (의도 분석 + 필터 추출, 서비스 경로 전용)
    → retrieve(): BM25 + Dense → RRF → 재순위 (실험과 같은 함수, 지시서 §5.6)
    → 상위 RERANK_TOP_N 단위를 출처로 변환 (헤더 + 대표 파트 본문)
    → 컨텍스트 조합 → LLM 응답 생성 → 출처 포함 결과 반환

보안 모드 OFF(외부 API)는 케이스 자료를 쓰지 않는 외부 조회다: 검색하지 않고, 컨텍스트 없이 질의만
외부 LLM 에 보내며, 출처를 붙이지 않는다 (v4.4 §6, 라우터도 컨텍스트를 한 번 더 차단).

서비스 후처리(min_source_length, rerank_min_score)는 평가 대상이 아니다. 실험은
retrieve() 결과를 그대로 평가한다.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from src.index import case_index
from src.llm.router import LLMRouter
from src.rag.query_parser import parse_query
from src.rag.retrieval import RetrievalResult, retrieve, service_config
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
    # v4 단위 정보 (선택)
    unit_id: str = ""
    page: int | None = None
    lang: str = ""


# 단위 유형 → 기존 source_type 이름 (프론트엔드 호환: 채팅은 teams_chat)
_UNIT_TO_SOURCE_TYPE = {"chat": "teams_chat"}


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
    # 토큰 사용량 (input_tokens/output_tokens/total_tokens/source). LLM 미호출 시 빈 dict.
    token_usage: dict[str, Any] = field(default_factory=dict)


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
    uncited = citation_count == 0 and not is_refusal(answer)
    return citation_count, invalid, uncited


def extract_citations(answer: str) -> list[int]:
    """응답의 [출처 N] 번호 목록 (나온 순서, 중복 포함) — validate_citations 와 같은 정규식"""
    return [int(m) for m in _CITATION_RE.findall(answer or "")]


def is_refusal(answer: str) -> bool:
    """'찾을 수 없습니다'류 거절 답변인지 (validate_citations 와 같은 패턴)"""
    return any(p in (answer or "") for p in _REFUSAL_PATTERNS)


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
        llm_router: LLMRouter | None = None,
        rerank_enabled: bool | None = None,
        retrieve_fn: Callable[..., RetrievalResult] | None = None,
    ) -> None:
        """RAGEngine 초기화

        Args:
            case_id: 검색 대상 케이스 ID
            llm_router: LLM 라우터 (기본: 새 인스턴스)
            rerank_enabled: Reranker 사용 여부 (None이면 settings.rerank_enabled)
            retrieve_fn: 검색 함수 (기본 retrieval.retrieve, 테스트 주입용)
        """
        self.case_id = case_id
        self.llm_router = llm_router or LLMRouter()
        self.rerank_enabled = (
            rerank_enabled if rerank_enabled is not None else settings.rerank_enabled
        )
        self._retrieve = retrieve_fn or retrieve
        self.last_retrieval: RetrievalResult | None = None

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        n_results: int | None = None,
    ) -> list[SourceReference]:
        """검색만 수행 (LLM 호출 없이) — retrieve() 상위 n개 단위를 출처로 변환

        재순위를 켠 경우 RERANK_MIN_SCORE(기본 0.0 = 끔) 미만 단위는 출처에서 뺀다.

        Args:
            query: 검색 쿼리
            filters: 메타데이터 필터 (source_type, _date_gte/_date_lte/_date_eq, participants 등)
            n_results: 결과 수 (기본: settings.rerank_top_n)

        Returns:
            SourceReference 리스트 (순위 순)

        Raises:
            IndexNotFoundError / IndexVersionError / EmbeddingDimensionError: 재인덱싱 필요
        """
        cfg = service_config(rerank=self.rerank_enabled)
        result = self._retrieve(self.case_id, query, cfg, filters)
        self.last_retrieval = result
        n = n_results or settings.rerank_top_n
        idx = case_index.load(self.case_id)
        method = "hybrid+rerank" if cfg.rerank else "hybrid"
        min_score = settings.rerank_min_score

        sources: list[SourceReference] = []
        for hit in result.hits:
            if len(sources) >= n:
                break
            if (
                cfg.rerank
                and min_score > 0
                and "rerank" in hit.stage_ranks
                and hit.score < min_score
            ):
                continue
            part = idx.part(hit.best_part_id)
            sources.append(_to_source(part, hit.score, method))
        logger.info(
            f"검색: case={self.case_id}, 단위 {len(result.hits)}건 중 {len(sources)}건 출처, "
            f"{result.timings_ms.get('total', 0):.0f}ms (재순위 절단 {result.n_rerank_truncated})"
        )
        return sources

    @staticmethod
    def _apply_relative_score_filter(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """상위 결과 평균 점수의 30% 미만인 결과 제거 (v3 상대 필터)

        v4 기본 경로에서는 호출하지 않는다(지시서 §5.7). 체크포인트 3에서 삭제한다.
        """
        if len(results) <= 1:
            return results
        scores = [r.get("score", 0.0) for r in results]
        mean_score = sum(scores) / len(scores)
        if mean_score <= 0:
            return results
        threshold = mean_score * 0.3
        filtered = [r for r in results if r.get("score", 0.0) >= threshold]
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
        if (
            settings.enable_relevance_check
            and filtered
            and not self._has_keyword_overlap(search_query, filtered)
        ):
            logger.warning(f"검색 결과와 질문의 키워드 일치도가 낮습니다 (case={self.case_id})")
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
        from src.units.topics import extract_topics

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

        # 보안 모드 OFF: 케이스 자료를 쓰지 않는 외부 조회 — 검색하지 않고 출처도 붙이지 않는다 (v4.4 §6)
        if not is_secure:
            return await self._external_query(question)

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
        token_usage: dict[str, Any] = {}

        def _capture_usage(rec: dict[str, Any]) -> None:
            token_usage.update(_usage_subset(rec))

        try:
            answer = await self.llm_router.generate(
                question=question,  # LLM에는 원본 질의 전달
                context=context_texts,
                secure_mode=is_secure,
                extra_system_warning=extra_warning,
                case_id=self.case_id,
                on_complete=_capture_usage,
            )
        except Exception as e:
            logger.error(f"LLM 응답 생성 실패: {e}")
            answer = f"LLM 응답 생성에 실패했습니다: {e}\n\n아래 검색 결과를 직접 확인해 주세요."

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
            logger.warning(f"LLM이 출처 인용 없이 답변 — 할루시네이션 위험 (case={self.case_id})")

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
            token_usage=token_usage,
        )

    async def query_stream(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        secure_mode: bool | None = None,
        n_results: int | None = None,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
        on_complete: Callable[[dict], None] | None = None,
    ) -> tuple[AsyncIterator[str], list[SourceReference]]:
        """스트리밍 RAG 질의: 검색 → LLM 스트리밍 응답

        Args:
            is_disconnected: 클라이언트 연결 끊김 여부를 반환하는 async 콜백 (선택).
                검색 시작 전과 LLM 스트림 시작 직전(=reranker 완료 후)에 체크하여,
                이미 끊겼으면 LLM을 호출하지 않고 빈 스트림을 반환한다. None이면 기존 동작.
            on_complete: 토큰 사용량 dict를 받는 완료 콜백 (선택). 스트리밍 종료 시
                generate_stream의 finally에서 호출된다.

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

        # 보안 모드 OFF: 케이스 자료를 쓰지 않는 외부 조회 — 검색하지 않고 출처도 붙이지 않는다 (v4.4 §6)
        if not is_secure:
            logger.info(f"외부 조회 (보안 모드 OFF, 케이스 자료 미사용): case={self.case_id}")
            return self.llm_router.generate_stream(
                question=question,
                context=[],
                secure_mode=False,
                case_id=self.case_id,
                on_complete=on_complete,
            ), []

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
            case_id=self.case_id,
            on_complete=on_complete,
        )

        return token_stream, sources

    async def _external_query(self, question: str) -> QueryResult:
        """보안 모드 OFF 질의 — 케이스 검색 없이 외부 LLM 에 질의만 보낸다 (출처 없음, v4.4 §6)"""
        token_usage: dict[str, Any] = {}

        def _capture_usage(rec: dict[str, Any]) -> None:
            token_usage.update(_usage_subset(rec))

        logger.info(f"외부 조회 (보안 모드 OFF, 케이스 자료 미사용): case={self.case_id}")
        try:
            answer = await self.llm_router.generate(
                question=question,
                context=[],
                secure_mode=False,
                case_id=self.case_id,
                on_complete=_capture_usage,
            )
        except Exception as e:
            logger.error(f"외부 LLM 응답 생성 실패: {e}")
            answer = f"외부 LLM 응답 생성에 실패했습니다: {e}"
        return QueryResult(
            answer=answer,
            sources=[],
            case_id=self.case_id,
            secure_mode=False,
            token_usage=token_usage,
        )

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
            header_parts.append(f"유형: {type_labels.get(source.source_type, source.source_type)}")
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


def _to_source(part: Any, score: float, method: str) -> SourceReference:
    """증거 단위 대표 파트 → SourceReference (content = 헤더 + 대표 파트 본문)"""
    meta = part.meta
    filename = meta.get("filename") or ""
    if not filename and part.unit_type in ("document", "attachment"):
        filename = str(meta.get("source_file", "")).rsplit("/", 1)[-1]
    if part.unit_type == "email":
        filename = filename or str(meta.get("source_file", "")).rsplit("/", 1)[-1]
    return SourceReference(
        content=part.full_text(with_header=True),
        source_type=_UNIT_TO_SOURCE_TYPE.get(part.unit_type, part.unit_type),
        filename=filename,
        date=str(meta.get("date") or ""),
        participants=list(meta.get("participants") or []),
        subject=str(meta.get("subject") or meta.get("parent_subject") or ""),
        score=float(score),
        search_method=method,
        sender=str(meta.get("sender") or ""),
        recipients=list(meta.get("recipients") or []),
        cc=list(meta.get("cc") or []),
        attachments=list(meta.get("attachment_filenames") or []),
        message_id=str(meta.get("message_id") or ""),
        in_reply_to=str(meta.get("in_reply_to") or ""),
        author=str(meta.get("author") or ""),
        last_modified_by=str(meta.get("last_modified_by") or ""),
        created_date=str(meta.get("created_date") or ""),
        last_modified=str(meta.get("last_modified") or ""),
        unit_id=part.unit_id,
        page=meta.get("page"),
        lang=str(meta.get("lang") or ""),
    )


def format_part_context(part: Any, score: float = 0.0, method: str = "") -> str:
    """증거 단위 대표 파트 → 서비스가 LLM 에 넣는 컨텍스트 문자열 (RAGEngine.query 와 같은 경로)

    실험 하네스(응답 생성 평가)가 서비스와 같은 컨텍스트를 쓰도록 공개한다 (v4.4 §4.1).
    """
    return RAGEngine._format_source_context(_to_source(part, score, method))


def _usage_subset(rec: dict[str, Any]) -> dict[str, Any]:
    """토큰 로깅 레코드에서 API/UI 노출용 부분집합만 추출"""
    return {
        "input_tokens": rec.get("input_tokens", 0),
        "output_tokens": rec.get("output_tokens", 0),
        "total_tokens": rec.get("total_tokens", 0),
        "source": rec.get("input_source", "estimated"),
    }


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
