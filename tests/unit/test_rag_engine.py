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


# === 컨텍스트 포맷 테스트 ===


class TestFormatContext:
    def test_format_email_source(self):
        """이메일 출처 포맷"""
        source = SourceReference(
            content="이메일 내용",
            source_type="email",
            filename="mail.eml",
            subject="회의록",
            date="2025-03-15",
            participants=["홍길동", "김철수"],
        )
        text = RAGEngine._format_source_context(source)
        assert "이메일" in text
        assert "회의록" in text
        assert "홍길동" in text
        assert "이메일 내용" in text

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

    def test_format_empty_source(self):
        """빈 출처 정보"""
        source = SourceReference(content="내용만", source_type="")
        text = RAGEngine._format_source_context(source)
        assert "내용만" in text
