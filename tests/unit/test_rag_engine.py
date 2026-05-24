"""RAG 엔진 유닛 테스트 — VectorStore + LLM 모킹"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.engine import RAGEngine, QueryResult, SourceReference


def _mock_search_results(n: int = 3) -> list[dict]:
    """모킹 검색 결과 생성"""
    return [
        {
            "content": f"검색 결과 {i} 내용입니다.",
            "metadata": {
                "source_type": "email" if i % 2 == 0 else "document",
                "filename": f"file_{i}.txt",
                "date": f"2025-01-{10 + i}",
                "participants": ["홍길동", "김철수"] if i % 2 == 0 else [],
                "subject": f"제목 {i}" if i % 2 == 0 else "",
            },
            "score": 0.9 - i * 0.1,
            "search_method": "hybrid",
        }
        for i in range(n)
    ]


@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.search.return_value = _mock_search_results()
    return vs


@pytest.fixture
def mock_llm_router():
    router = AsyncMock()
    router.generate.return_value = "LLM이 생성한 답변입니다."
    return router


@pytest.fixture
def engine(mock_vector_store, mock_llm_router):
    return RAGEngine(
        case_id="test_case",
        vector_store=mock_vector_store,
        llm_router=mock_llm_router,
    )


# === 검색 테스트 ===


class TestSearch:
    def test_search_returns_sources(self, engine):
        """검색 결과가 SourceReference로 변환"""
        sources = engine.search("비용 분석")
        assert len(sources) == 3
        assert all(isinstance(s, SourceReference) for s in sources)

    def test_search_metadata_mapping(self, engine):
        """검색 결과 메타데이터 매핑"""
        sources = engine.search("테스트")
        assert sources[0].source_type == "email"
        assert sources[0].filename == "file_0.txt"
        assert sources[0].score == 0.9
        assert "홍길동" in sources[0].participants
        assert sources[0].subject == "제목 0"

    def test_search_empty_results(self, engine, mock_vector_store):
        """검색 결과 없음"""
        mock_vector_store.search.return_value = []
        sources = engine.search("존재하지 않는 내용")
        assert sources == []

    def test_search_passes_filters(self, engine, mock_vector_store):
        """필터가 벡터 저장소에 전달되는지 확인"""
        engine.search("질의", filters={"source_type": "email"})
        mock_vector_store.search.assert_called_once_with(
            query="질의",
            n_results=engine.top_k,
            filters={"source_type": "email"},
        )


# === 질의 테스트 ===


class TestQuery:
    @pytest.mark.asyncio
    async def test_query_full_flow(self, engine, mock_llm_router):
        """전체 질의 흐름: 검색 → LLM → 결과"""
        result = await engine.query("비용 관련 내용은?")

        assert isinstance(result, QueryResult)
        assert result.answer == "LLM이 생성한 답변입니다."
        assert len(result.sources) == 3
        assert result.case_id == "test_case"
        mock_llm_router.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_no_results(self, engine, mock_vector_store):
        """검색 결과 없을 때"""
        mock_vector_store.search.return_value = []
        result = await engine.query("아무것도 없는 질의")

        assert "검색 결과가 없습니다" in result.answer
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_query_llm_error_fallback(self, engine, mock_llm_router):
        """LLM 호출 실패 시 에러 메시지 + 검색 결과는 유지"""
        mock_llm_router.generate.side_effect = Exception("connection refused")
        result = await engine.query("질문")

        assert "실패" in result.answer
        assert len(result.sources) == 3  # 검색 결과는 유지

    @pytest.mark.asyncio
    async def test_query_passes_secure_mode(self, engine, mock_llm_router):
        """보안 모드가 LLM에 전달되는지 확인"""
        await engine.query("질문", secure_mode=False)
        _, kwargs = mock_llm_router.generate.call_args
        assert kwargs["secure_mode"] is False

    @pytest.mark.asyncio
    async def test_query_passes_filters(self, engine, mock_vector_store):
        """필터가 검색에 전달되는지 확인"""
        await engine.query("질문", filters={"source_type": "email"})
        mock_vector_store.search.assert_called_once()
        _, kwargs = mock_vector_store.search.call_args
        assert kwargs["filters"] == {"source_type": "email"}


# === 스트리밍 중단 (query_stream is_disconnected) ===


async def _collect_stream(agen) -> list[str]:
    """async generator의 모든 토큰을 리스트로 수집"""
    out = []
    async for t in agen:
        out.append(t)
    return out


class TestQueryStreamDisconnect:
    def _stream_engine(self, mock_vector_store):
        """generate_stream이 async generator를 반환하는 엔진"""
        router = MagicMock()

        async def fake_stream(**kwargs):
            for tok in ["토큰1 ", "토큰2 ", "토큰3"]:
                yield tok

        router.generate_stream = fake_stream
        return RAGEngine(
            case_id="test_case",
            vector_store=mock_vector_store,
            llm_router=router,
        )

    @pytest.mark.asyncio
    async def test_normal_stream_without_callback(self, mock_vector_store):
        """is_disconnected 미전달 시 기존 동작 — 토큰 정상 생성"""
        engine = self._stream_engine(mock_vector_store)
        token_stream, sources = await engine.query_stream("질문")
        tokens = await _collect_stream(token_stream)
        assert "".join(tokens) == "토큰1 토큰2 토큰3"
        assert len(sources) == 3

    @pytest.mark.asyncio
    async def test_disconnect_before_search(self, mock_vector_store):
        """검색 전에 이미 끊김 → 빈 스트림 + 빈 sources, LLM 호출 안 함"""
        engine = self._stream_engine(mock_vector_store)

        async def always_disconnected() -> bool:
            return True

        token_stream, sources = await engine.query_stream(
            "질문", is_disconnected=always_disconnected
        )
        tokens = await _collect_stream(token_stream)
        assert tokens == []
        assert sources == []
        # 검색조차 호출 안 됨
        mock_vector_store.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_after_search_before_llm(self, mock_vector_store):
        """검색 후 LLM 직전 끊김 → 빈 스트림 + sources는 반환"""
        engine = self._stream_engine(mock_vector_store)

        calls = {"n": 0}

        async def disconnect_second_call() -> bool:
            # 첫 체크(검색 전)는 False, 두 번째(LLM 전)는 True
            calls["n"] += 1
            return calls["n"] >= 2

        token_stream, sources = await engine.query_stream(
            "질문", is_disconnected=disconnect_second_call
        )
        tokens = await _collect_stream(token_stream)
        assert tokens == []  # LLM 스트림 시작 안 함
        assert len(sources) == 3  # 검색은 완료됐으므로 sources 있음
        mock_vector_store.search.assert_called_once()


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
    def test_rerank_disabled_by_default(self, mock_vector_store, mock_llm_router):
        """rerank_enabled 기본값은 settings 기반 (기본 False)"""
        engine = RAGEngine(
            case_id="test",
            vector_store=mock_vector_store,
            llm_router=mock_llm_router,
        )
        assert engine.rerank_enabled is False

    def test_rerank_disabled_uses_normal_search(self, mock_vector_store, mock_llm_router):
        """rerank OFF → 기존 검색 그대로"""
        engine = RAGEngine(
            case_id="test",
            vector_store=mock_vector_store,
            llm_router=mock_llm_router,
            rerank_enabled=False,
        )
        sources = engine.search("질의")
        mock_vector_store.search.assert_called_once()
        assert len(sources) == 3

    @patch("src.rag.reranker.get_reranker")
    def test_rerank_enabled_calls_reranker(
        self, mock_get_reranker, mock_vector_store, mock_llm_router
    ):
        """rerank ON → reranker 호출"""
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {
                "content": "reranked 내용",
                "metadata": {"source_type": "email", "filename": "r.eml"},
                "score": 0.5,
                "rerank_score": 0.95,
                "search_method": "hybrid",
            }
        ]
        mock_get_reranker.return_value = mock_reranker

        engine = RAGEngine(
            case_id="test",
            vector_store=mock_vector_store,
            llm_router=mock_llm_router,
            rerank_enabled=True,
        )
        sources = engine.search("질의")

        # 후보 수 50개로 검색
        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs["n_results"] == 50

        # reranker 호출됨
        mock_reranker.rerank.assert_called_once()

        # rerank_score가 score로 매핑됨
        assert len(sources) == 1
        assert sources[0].score == pytest.approx(0.95)

    def test_rerank_enabled_empty_results(self, mock_vector_store, mock_llm_router):
        """rerank ON + 검색 결과 없음 → reranker 호출 안 됨"""
        mock_vector_store.search.return_value = []
        engine = RAGEngine(
            case_id="test",
            vector_store=mock_vector_store,
            llm_router=mock_llm_router,
            rerank_enabled=True,
        )
        sources = engine.search("질의")
        assert sources == []

    @pytest.mark.asyncio
    @patch("src.rag.reranker.get_reranker")
    async def test_query_with_rerank(
        self, mock_get_reranker, mock_vector_store, mock_llm_router
    ):
        """전체 query 흐름에서 rerank 동작"""
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = _mock_search_results(2)
        for doc in mock_reranker.rerank.return_value:
            doc["rerank_score"] = 0.9
        mock_get_reranker.return_value = mock_reranker

        engine = RAGEngine(
            case_id="test",
            vector_store=mock_vector_store,
            llm_router=mock_llm_router,
            rerank_enabled=True,
        )
        result = await engine.query("질문")

        assert len(result.sources) == 2
        mock_llm_router.generate.assert_called_once()
