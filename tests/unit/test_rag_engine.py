"""RAG 엔진 유닛 테스트 — retrieve()·케이스 색인·LLM 모킹 (v4)"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.index import case_index
from src.index.case_index import ProfileVectors
from src.rag.engine import QueryResult, RAGEngine, SourceReference, validate_citations
from src.rag.retrieval import RetrievalConfig, RetrievalResult, UnitHit
from src.units.schema import UnitPart
from src.utils.config import settings

CASE = "test_case"


def _parts() -> list[UnitPart]:
    """이메일 2 + 문서 1 (본문은 min_source_length=20자 이상)"""
    out = []
    for i in range(3):
        is_email = i % 2 == 0
        meta = {
            "case_id": CASE,
            "source_file": f"box/file_{i}.{'eml' if is_email else 'txt'}",
            "date": f"2025-01-{10 + i}T09:00:00+09:00",
            "participants": ["홍길동", "김철수"] if is_email else [],
            "subject": f"제목 {i}" if is_email else "",
            "sender": "홍길동 <hong@x.com>" if is_email else "",
            "lang": "ko",
        }
        if not is_email:
            meta.update({"filename": f"file_{i}.txt", "page": 1, "author": "alice"})
        uid = f"em:{i}" if is_email else f"doc:{i}:p1"
        out.append(UnitPart(
            f"{uid}#1", uid, "email" if is_email else "document",
            f"[{'이메일' if is_email else '문서'}] 헤더 {i}",
            f"검색 결과 {i} 내용입니다. 충분히 긴 본문 텍스트를 포함하고 있습니다.", 20, meta,
        ))
    return out


def _write_index(parts: list[UnitPart]) -> None:
    vecs = np.ones((len(parts), 4), dtype=np.float32) / 2
    case_index.write_index(CASE, parts, {"hdr-qwen3": ProfileVectors(vecs, settings.ollama_embed_model)})


def _result(parts: list[UnitPart], scores: list[float] | None = None, rerank: bool = False) -> RetrievalResult:
    scores = scores or [0.9 - i * 0.1 for i in range(len(parts))]
    hits = []
    for r, (p, s) in enumerate(zip(parts, scores, strict=True), 1):
        ranks = {"bm25": r, "dense": r, "rrf": r}
        if rerank:
            ranks["rerank"] = r
        hits.append(UnitHit(p.unit_id, r, s, p.part_id, ranks))
    return RetrievalResult(hits=hits, timings_ms={"total": 1.0}, n_rerank_truncated=0)


@pytest.fixture(autouse=True)
def _tmp_index(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "index_dir", str(tmp_path / "index"))
    monkeypatch.setattr(settings, "rerank_min_score", 0.0)
    case_index._cache.clear()
    _write_index(_parts())
    yield
    case_index._cache.clear()


@pytest.fixture
def mock_retrieve():
    return MagicMock(return_value=_result(_parts()))


@pytest.fixture
def mock_llm_router():
    router = AsyncMock()
    router.generate.return_value = "LLM이 생성한 답변입니다."
    return router


@pytest.fixture
def engine(mock_retrieve, mock_llm_router):
    return RAGEngine(
        case_id=CASE,
        llm_router=mock_llm_router,
        rerank_enabled=False,
        retrieve_fn=mock_retrieve,
    )


# === 검색 테스트 ===


class TestSearch:
    def test_search_returns_sources(self, engine):
        """retrieve() 단위가 SourceReference로 변환"""
        sources = engine.search("비용 분석")
        assert len(sources) == 3
        assert all(isinstance(s, SourceReference) for s in sources)

    def test_search_metadata_mapping(self, engine):
        """단위 메타 → 출처 필드, content = 헤더 + 대표 파트 본문"""
        sources = engine.search("테스트")
        s0, s1 = sources[0], sources[1]
        assert s0.source_type == "email" and s0.unit_id == "em:0"
        assert s0.filename == "file_0.eml"
        assert s0.score == pytest.approx(0.9)
        assert "홍길동" in s0.participants and s0.subject == "제목 0"
        assert s0.content.startswith("[이메일] 헤더 0\n\n검색 결과 0")
        assert s1.source_type == "document" and s1.filename == "file_1.txt"
        assert s1.page == 1 and s1.author == "alice" and s1.lang == "ko"

    def test_chat_unit_maps_to_teams_chat(self, mock_llm_router):
        """채팅 단위는 프론트엔드 호환 이름 teams_chat"""
        part = UnitPart("ch:a:1#1", "ch:a:1", "chat", "[채팅] 방", "10:00 홍길동: 송금 완료 보고드립니다 확인 부탁", 9,
                        {"participants": ["홍길동"]})
        _write_index([part])
        eng = RAGEngine(CASE, llm_router=mock_llm_router, rerank_enabled=False,
                        retrieve_fn=MagicMock(return_value=_result([part])))
        assert eng.search("q")[0].source_type == "teams_chat"

    def test_search_top_n(self, engine, monkeypatch):
        """RERANK_TOP_N개까지만 출처로"""
        monkeypatch.setattr(settings, "rerank_top_n", 2)
        assert len(engine.search("q")) == 2
        assert len(engine.search("q", n_results=1)) == 1

    def test_search_empty_results(self, engine, mock_retrieve):
        mock_retrieve.return_value = RetrievalResult([], {"total": 0.0}, 0)
        assert engine.search("존재하지 않는 내용") == []

    def test_search_passes_filters_and_service_config(self, engine, mock_retrieve):
        """필터와 서비스 설정이 retrieve()에 전달된다"""
        engine.search("질의", filters={"source_type": "email"})
        args = mock_retrieve.call_args.args
        assert args[0] == CASE and args[1] == "질의"
        assert isinstance(args[2], RetrievalConfig) and args[2].rerank is False
        assert args[3] == {"source_type": "email"}

    def test_rerank_min_score_only_when_set(self, mock_llm_router, monkeypatch):
        """RERANK_MIN_SCORE 기본 0.0이면 거르지 않고, 설정하면 재순위 구간만 거른다"""
        parts = _parts()
        rr = MagicMock(return_value=_result(parts, [0.9, 0.05, 0.01], rerank=True))
        eng = RAGEngine(CASE, llm_router=mock_llm_router, rerank_enabled=True, retrieve_fn=rr)
        assert len(eng.search("q")) == 3
        monkeypatch.setattr(settings, "rerank_min_score", 0.1)
        assert [s.unit_id for s in eng.search("q")] == ["em:0"]

    def test_missing_index_raises(self, engine):
        """색인이 없으면 재인덱싱 필요 오류 (조용히 빈 결과를 내지 않음)"""
        case_index.delete(CASE)
        with pytest.raises(case_index.IndexNotFoundError):
            engine.search("q")


# === 질의 테스트 ===


class TestQuery:
    @pytest.mark.asyncio
    async def test_query_full_flow(self, engine, mock_llm_router):
        """전체 질의 흐름: 검색 → LLM → 결과"""
        result = await engine.query("비용 관련 내용은?")

        assert isinstance(result, QueryResult)
        assert result.answer == "LLM이 생성한 답변입니다."
        assert len(result.sources) == 3
        assert result.case_id == CASE
        mock_llm_router.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_no_results(self, engine, mock_retrieve):
        mock_retrieve.return_value = RetrievalResult([], {"total": 0.0}, 0)
        result = await engine.query("아무것도 없는 질의")
        assert "검색 결과가 없습니다" in result.answer
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_query_llm_error_fallback(self, engine, mock_llm_router):
        """LLM 호출 실패 시 에러 메시지 + 검색 결과는 유지"""
        mock_llm_router.generate.side_effect = Exception("connection refused")
        result = await engine.query("질문")
        assert "실패" in result.answer
        assert len(result.sources) == 3

    @pytest.mark.asyncio
    async def test_query_passes_secure_mode(self, engine, mock_llm_router):
        await engine.query("질문", secure_mode=False)
        _, kwargs = mock_llm_router.generate.call_args
        assert kwargs["secure_mode"] is False

    @pytest.mark.asyncio
    async def test_query_passes_filters(self, engine, mock_retrieve):
        await engine.query("질문", filters={"source_type": "email"})
        mock_retrieve.assert_called_once()
        assert mock_retrieve.call_args.args[3] == {"source_type": "email"}


# === 스트리밍 중단 (query_stream is_disconnected) ===


async def _collect_stream(agen) -> list[str]:
    """async generator의 모든 토큰을 리스트로 수집"""
    out = []
    async for t in agen:
        out.append(t)
    return out


class TestQueryStreamDisconnect:
    def _stream_engine(self, mock_retrieve):
        """generate_stream이 async generator를 반환하는 엔진"""
        router = MagicMock()

        async def fake_stream(**kwargs):
            for tok in ["토큰1 ", "토큰2 ", "토큰3"]:
                yield tok

        router.generate_stream = fake_stream
        return RAGEngine(case_id=CASE, llm_router=router, rerank_enabled=False, retrieve_fn=mock_retrieve)

    @pytest.mark.asyncio
    async def test_normal_stream_without_callback(self, mock_retrieve):
        engine = self._stream_engine(mock_retrieve)
        token_stream, sources = await engine.query_stream("질문")
        tokens = await _collect_stream(token_stream)
        assert "".join(tokens) == "토큰1 토큰2 토큰3"
        assert len(sources) == 3

    @pytest.mark.asyncio
    async def test_disconnect_before_search(self, mock_retrieve):
        """검색 전에 이미 끊김 → 빈 스트림 + 빈 sources, 검색도 안 함"""
        engine = self._stream_engine(mock_retrieve)

        async def always_disconnected() -> bool:
            return True

        token_stream, sources = await engine.query_stream("질문", is_disconnected=always_disconnected)
        assert await _collect_stream(token_stream) == []
        assert sources == []
        mock_retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_after_search_before_llm(self, mock_retrieve):
        """검색 후 LLM 직전 끊김 → 빈 스트림 + sources는 반환"""
        engine = self._stream_engine(mock_retrieve)
        calls = {"n": 0}

        async def disconnect_second_call() -> bool:
            calls["n"] += 1
            return calls["n"] >= 2

        token_stream, sources = await engine.query_stream("질문", is_disconnected=disconnect_second_call)
        assert await _collect_stream(token_stream) == []
        assert len(sources) == 3
        mock_retrieve.assert_called_once()

# === 컨텍스트 포맷 테스트 ===


class TestFormatContext:
    def test_format_email_source(self):
        """이메일 출처 포맷 — sender/recipients/cc/첨부 분리 표시"""
        source = SourceReference(
            content="이메일 내용",
            source_type="email",
            filename="mail.eml",
            subject="회의록",
            date="2025-03-15",
            participants=["홍길동", "김철수"],  # legacy fallback (이메일에선 미사용)
            sender="홍길동 <hong@example.com>",
            recipients=["김철수 <kim@example.com>"],
            cc=["박영희 <park@example.com>"],
            attachments=["보고서.pdf"],
        )
        text = RAGEngine._format_source_context(source)
        assert "이메일" in text
        assert "회의록" in text
        assert "이메일 내용" in text
        # 이메일 분리 표시
        assert "From:" in text and "홍길동" in text
        assert "To:" in text and "김철수" in text
        assert "Cc:" in text and "박영희" in text
        assert "첨부:" in text and "보고서.pdf" in text

    def test_format_document_source(self):
        """문서 출처 포맷"""
        source = SourceReference(
            content="문서 내용",
            source_type="document",
            filename="report.pdf",
        )
        text = RAGEngine._format_source_context(source)
        assert "문서" in text
        assert "report.pdf" in text

    def test_format_document_with_author(self):
        """문서 출처 — author/last_modified_by/created_date/last_modified 표시"""
        source = SourceReference(
            content="견적서 내용",
            source_type="document",
            filename="quote.xlsx",
            author="alice",
            last_modified_by="bob",
            created_date="2026-03-01T10:00:00",
            last_modified="2026-04-15T16:30:00",
        )
        text = RAGEngine._format_source_context(source)
        assert "작성자: alice" in text
        assert "마지막 수정: bob" in text
        assert "생성일: 2026-03-01T10:00:00" in text
        assert "수정일: 2026-04-15T16:30:00" in text

    def test_format_document_same_author_modifier(self):
        """문서 출처 — author == last_modified_by이면 수정자 라인 생략"""
        source = SourceReference(
            content="x",
            source_type="document",
            filename="own.docx",
            author="alice",
            last_modified_by="alice",
            created_date="2026-03-01",
            last_modified="2026-03-01",
        )
        text = RAGEngine._format_source_context(source)
        assert "작성자: alice" in text
        assert "마지막 수정" not in text  # 동일하므로 생략
        assert "수정일" not in text  # created_date == last_modified이면 생략

    def test_format_empty_source(self):
        """빈 출처 정보"""
        source = SourceReference(content="내용만", source_type="")
        text = RAGEngine._format_source_context(source)
        assert "내용만" in text



# === Reranker 통합 테스트 ===


class TestRerankIntegration:
    @patch("src.rag.engine.settings")
    def test_rerank_enabled_by_default(self, mock_settings, mock_llm_router):
        """rerank_enabled 기본값은 settings 기반 (.env 영향 배제를 위해 settings mock)"""
        mock_settings.rerank_enabled = True
        engine = RAGEngine(case_id="test", llm_router=mock_llm_router)
        assert engine.rerank_enabled is True

    def test_rerank_flag_goes_to_retrieval_config(self, mock_retrieve, mock_llm_router):
        """요청별 rerank 켜고 끄기 → RetrievalConfig.rerank"""
        for flag in (True, False):
            eng = RAGEngine(CASE, llm_router=mock_llm_router, rerank_enabled=flag, retrieve_fn=mock_retrieve)
            eng.search("질의")
            assert mock_retrieve.call_args.args[2].rerank is flag

    def test_service_config_uses_settings(self, mock_retrieve, mock_llm_router, monkeypatch):
        """서비스 경로 프로필·백엔드·후보 수는 settings 값"""
        monkeypatch.setattr(settings, "rerank_backend", "qwen3")
        monkeypatch.setattr(settings, "rerank_candidates", 30)
        eng = RAGEngine(CASE, llm_router=mock_llm_router, rerank_enabled=True, retrieve_fn=mock_retrieve)
        eng.search("질의")
        cfg = mock_retrieve.call_args.args[2]
        assert (cfg.profile, cfg.reranker, cfg.rerank_candidates, cfg.k_bm25) == ("hdr-qwen3", "qwen3", 30, 50)

    def test_end_to_end_fake_models(self, mock_llm_router, monkeypatch):
        """가짜 모델로 실제 retrieve() + 재순위 경로"""
        from src.embeddings.embedding_service import fake_embed
        from src.index.case_index import embed_input
        from src.rag import retrieval
        from src.rag.reranker import release_rerankers

        monkeypatch.setattr(settings, "ast_fake_models", True)
        monkeypatch.setattr(settings, "embed_dim", 16)
        monkeypatch.setattr(retrieval, "_embedders", {})
        release_rerankers()
        parts = _parts()
        parts[1].text = "협력업체 선급금 송금 결재 누락 정황이 있습니다. 확인 바랍니다."
        vecs = np.array([fake_embed(embed_input(p, True), 16) for p in parts])
        case_index.write_index(CASE, parts, {"hdr-qwen3": ProfileVectors(vecs, settings.ollama_embed_model)})
        eng = RAGEngine(CASE, llm_router=mock_llm_router, rerank_enabled=True)
        sources = eng.search("선급금 결재")
        assert sources[0].unit_id == "doc:1:p1"
        assert sources[0].search_method == "hybrid+rerank"
        assert eng.last_retrieval is not None and "rerank" in eng.last_retrieval.timings_ms
        release_rerankers()

# === 할루시네이션 제어 (#2~#5) ===


class TestValidateCitations:
    def test_valid_citations(self):
        cc, invalid, uncited = validate_citations("비용 증가 [출처 1][출처 2].", 3)
        assert cc == 2
        assert invalid == []
        assert uncited is False

    def test_invalid_citation_out_of_range(self):
        cc, invalid, uncited = validate_citations("[출처 7]에 따르면", 5)
        assert cc == 1
        assert invalid == [7]
        assert uncited is False

    def test_uncited_response(self):
        cc, invalid, uncited = validate_citations("그냥 답변입니다 근거 없음.", 3)
        assert cc == 0
        assert uncited is True

    def test_refusal_not_flagged_as_uncited(self):
        """거절 답변은 인용 없어도 uncited 아님"""
        cc, invalid, uncited = validate_citations(
            "제공된 자료에서 해당 내용을 찾을 수 없습니다.", 3
        )
        assert cc == 0
        assert uncited is False

    def test_citation_without_space(self):
        cc, invalid, uncited = validate_citations("내용 [출처3]", 5)
        assert cc == 1
        assert invalid == []


class TestShortChunkFilter:
    def _engine(self, texts: list[str], mock_llm_router):
        parts = [
            UnitPart(f"doc:{i}:p1#1", f"doc:{i}:p1", "document", "", t, 1, {"filename": f"{i}.txt"})
            for i, t in enumerate(texts)
        ]
        _write_index(parts)
        return RAGEngine(CASE, llm_router=mock_llm_router, rerank_enabled=False,
                         retrieve_fn=MagicMock(return_value=_result(parts)))

    @pytest.mark.asyncio
    async def test_short_sources_removed(self, mock_llm_router):
        """20자 미만 소스 제거 후 LLM 호출"""
        eng = self._engine(["승인", "이것은 충분히 긴 본문 텍스트입니다. 의미 있는 내용을 담고 있습니다."], mock_llm_router)
        result = await eng.query("질문")
        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_all_short_returns_no_results(self, mock_llm_router):
        """전부 짧으면 검색 결과 없음 처리"""
        eng = self._engine(["확인", "승인"], mock_llm_router)
        result = await eng.query("질문")
        assert result.sources == []
        assert "검색 결과가 없습니다" in result.answer
        mock_llm_router.generate.assert_not_called()


class TestCitationInQueryResult:
    @pytest.mark.asyncio
    async def test_query_result_has_citation_meta(self, engine, mock_llm_router):
        """query 결과에 citation 메타 포함"""
        mock_llm_router.generate.return_value = "답변입니다 [출처 1]. 추가 [출처 9]."
        result = await engine.query("질문")
        assert result.citation_count == 2
        assert 9 in result.invalid_citations
        assert result.uncited_response is False


class TestRelativeScoreFilter:
    def test_low_score_removed(self):
        results = [
            {"content": "a", "score": 1.0},
            {"content": "b", "score": 0.9},
            {"content": "c", "score": 0.05},  # 평균 대비 매우 낮음
        ]
        filtered = RAGEngine._apply_relative_score_filter(results)
        contents = [r["content"] for r in filtered]
        assert "c" not in contents

    def test_single_result_not_filtered(self):
        results = [{"content": "a", "score": 0.001}]
        assert RAGEngine._apply_relative_score_filter(results) == results


class TestRelevanceCheck:
    def test_keyword_overlap_true(self):
        sources = [SourceReference(content="비용이 15% 증가했습니다", source_type="document")]
        assert RAGEngine._has_keyword_overlap("비용 증가", sources) is True

    def test_keyword_overlap_false(self):
        sources = [SourceReference(content="날씨가 맑고 화창합니다", source_type="document")]
        # extract_topics는 10자 미만 질의를 무시하므로 충분히 긴 질의 사용
        assert RAGEngine._has_keyword_overlap("비용 예산 회계 처리 내역 검토", sources) is False


# === 보안 모드 OFF = 케이스 자료를 쓰지 않는 외부 조회 (v4.4 §6) ===


class TestExternalLookup:
    @pytest.mark.asyncio
    async def test_query_off_skips_retrieval_and_context(self, engine, mock_retrieve, mock_llm_router):
        result = await engine.query("공개 정보 질문", secure_mode=False)
        mock_retrieve.assert_not_called()
        _, kwargs = mock_llm_router.generate.call_args
        assert kwargs["context"] == [] and kwargs["secure_mode"] is False
        assert result.sources == [] and result.secure_mode is False

    @pytest.mark.asyncio
    async def test_query_stream_off_skips_retrieval_and_context(self, mock_retrieve):
        router = MagicMock()
        router.generate_stream = MagicMock(return_value="stream")
        eng = RAGEngine(case_id=CASE, llm_router=router, rerank_enabled=False, retrieve_fn=mock_retrieve)
        stream, sources = await eng.query_stream("공개 정보 질문", secure_mode=False)
        mock_retrieve.assert_not_called()
        assert sources == [] and stream == "stream"
        _, kwargs = router.generate_stream.call_args
        assert kwargs["context"] == [] and kwargs["secure_mode"] is False

    @pytest.mark.asyncio
    async def test_case_text_never_reaches_external_client(self, mock_retrieve):
        """실제 라우터 + 외부 클라이언트 대역: 색인 본문이 메시지에 없다"""
        from src.llm.router import LLMRouter

        sent: list = []

        class _Ext:
            async def ainvoke(self, messages):
                sent.append(messages)
                return MagicMock(content="외부 답변", usage_metadata=None, response_metadata={})

        router = LLMRouter()
        eng = RAGEngine(case_id=CASE, llm_router=router, rerank_enabled=False, retrieve_fn=mock_retrieve)
        with patch.object(router, "get_llm", return_value=_Ext()):
            await eng.query("검색 결과 0 내용", secure_mode=False)
        text = "\n".join(c for msgs in sent for _r, c in msgs)
        assert "충분히 긴 본문" not in text and "헤더 0" not in text
