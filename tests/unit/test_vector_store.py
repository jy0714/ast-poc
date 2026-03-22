"""VectorStoreService 유닛 테스트

ChromaDB + BM25 하이브리드 검색, 메타데이터 필터, 통계 검증.
임베딩은 모킹하여 Ollama 없이도 테스트 가능.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.chunkers.chunker import Chunk
from src.vectorstore.vector_store import VectorStoreService


@pytest.fixture
def mock_embedding():
    """EmbeddingService 모킹 — 고정 차원 벡터 반환"""
    dim = 8

    def fake_embed_text(text: str) -> list[float]:
        # 텍스트 해시 기반 결정적 벡터 생성
        import hashlib

        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i : i + 2], 16) / 255.0 for i in range(0, dim * 2, 2)]

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [fake_embed_text(t) for t in texts]

    mock = MagicMock()
    mock.embed_text.side_effect = fake_embed_text
    mock.embed_texts.side_effect = fake_embed_texts
    return mock


_collection_counter = 0


@pytest.fixture
def store(mock_embedding):
    """모킹된 임베딩을 사용하는 VectorStoreService (인메모리, 테스트별 격리)"""
    global _collection_counter
    _collection_counter += 1
    collection_name = f"test_col_{_collection_counter}"

    with tempfile.TemporaryDirectory() as bm25_dir:
        with patch("src.utils.config.settings") as mock_settings:
            mock_settings.bm25_index_dir = bm25_dir
            mock_settings.embed_batch_size = 50
            mock_settings.search_top_k = 10
            mock_settings.rrf_k = 60

            svc = VectorStoreService(
                collection_name=collection_name,
                ephemeral=True,
            )
            svc._embedding_service = mock_embedding
            yield svc


def _make_chunks(texts: list[str], source_type: str = "document", **extra_meta) -> list[Chunk]:
    """테스트용 청크 생성"""
    return [
        Chunk(
            content=text,
            metadata={"filename": f"doc{i}.pdf", **extra_meta},
            source_type=source_type,
        )
        for i, text in enumerate(texts)
    ]


# === 기본 저장/조회 테스트 ===


class TestAddChunks:
    def test_add_and_count(self, store):
        """청크 추가 후 카운트 확인"""
        chunks = _make_chunks(["감사 보고서 내용", "비용 분석 결과", "회의록 요약"])
        added = store.add_chunks(chunks)
        assert added == 3
        stats = store.get_stats()
        assert stats["total_chunks"] == 3

    def test_add_empty_list(self, store):
        """빈 리스트 추가"""
        assert store.add_chunks([]) == 0

    def test_duplicate_chunks_skipped(self, store):
        """중복 청크 스킵"""
        chunks = _make_chunks(["감사 보고서 내용"])
        store.add_chunks(chunks)
        # 동일 청크 다시 추가
        added = store.add_chunks(chunks)
        assert added == 0
        assert store.get_stats()["total_chunks"] == 1

    def test_source_type_in_metadata(self, store):
        """source_type이 메타데이터에 저장"""
        chunks = _make_chunks(["이메일 내용"], source_type="email")
        store.add_chunks(chunks)
        stats = store.get_stats()
        assert stats["source_type_counts"].get("email") == 1


# === 벡터 검색 테스트 ===


class TestVectorSearch:
    def test_basic_vector_search(self, store):
        """기본 벡터 검색"""
        chunks = _make_chunks([
            "감사 보고서에 따르면 비용이 증가했습니다",
            "Teams 채팅에서 회의 일정을 논의했습니다",
            "분기별 매출 분석 보고서입니다",
        ])
        store.add_chunks(chunks)

        results = store.search("감사 보고서", n_results=2, search_method="vector")
        assert len(results) <= 2
        assert all("content" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all("score" in r for r in results)
        assert all(r["search_method"] == "vector" for r in results)

    def test_vector_search_with_filter(self, store):
        """메타데이터 필터링 벡터 검색"""
        email_chunks = _make_chunks(["이메일 감사 내용"], source_type="email")
        doc_chunks = _make_chunks(["문서 감사 내용"], source_type="document")
        store.add_chunks(email_chunks)
        store.add_chunks(doc_chunks)

        results = store.search(
            "감사", n_results=5, filters={"source_type": "email"}, search_method="vector"
        )
        for r in results:
            assert r["metadata"].get("source_type") == "email"


# === BM25 검색 테스트 ===


class TestBM25Search:
    def test_basic_bm25_search(self, store):
        """기본 BM25 키워드 검색"""
        chunks = _make_chunks([
            "감사 보고서 비용 분석 결과를 정리했습니다",
            "오늘 점심 메뉴를 정했습니다",
            "감사팀 회의에서 비용 절감 방안을 논의했습니다",
        ])
        store.add_chunks(chunks)

        results = store.search("감사 비용", n_results=3, search_method="bm25")
        assert len(results) > 0
        # "감사"와 "비용"이 포함된 청크가 상위에 와야 함
        top_content = results[0]["content"]
        assert "감사" in top_content or "비용" in top_content

    def test_bm25_empty_store(self, store):
        """빈 저장소에서 BM25 검색"""
        results = store.search("테스트", search_method="bm25")
        assert results == []


# === 하이브리드 검색 테스트 ===


class TestHybridSearch:
    def test_hybrid_search(self, store):
        """하이브리드 검색 (RRF)"""
        chunks = _make_chunks([
            "감사 보고서에 따르면 비용이 증가했습니다",
            "Teams 채팅에서 회의 일정을 논의했습니다",
            "분기별 매출 분석 보고서입니다",
        ])
        store.add_chunks(chunks)

        results = store.search("감사 보고서", n_results=2)
        assert len(results) <= 2
        assert all(r["search_method"] == "hybrid" for r in results)
        # RRF 스코어가 있어야 함
        assert all(r["score"] > 0 for r in results)

    def test_hybrid_scores_ordered(self, store):
        """하이브리드 검색 결과가 스코어 내림차순"""
        chunks = _make_chunks([
            "감사 보고서 비용 분석",
            "감사 결과 보고",
            "오늘 날씨가 좋습니다",
            "내일 회의 일정",
        ])
        store.add_chunks(chunks)

        results = store.search("감사 보고서", n_results=4)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# === 컬렉션 관리 테스트 ===


class TestCollectionManagement:
    def test_delete_collection(self, store):
        """컬렉션 삭제"""
        chunks = _make_chunks(["테스트 데이터"])
        store.add_chunks(chunks)
        assert store.get_stats()["total_chunks"] == 1

        store.delete_collection()
        # 삭제 후 새 컬렉션은 비어 있어야 함
        stats = store.get_stats()
        assert stats["total_chunks"] == 0

    def test_get_stats(self, store):
        """통계 조회"""
        email_chunks = _make_chunks(["이메일 1", "이메일 2"], source_type="email")
        doc_chunks = _make_chunks(["문서 1"], source_type="document")
        store.add_chunks(email_chunks)
        store.add_chunks(doc_chunks)

        stats = store.get_stats()
        assert stats["collection_name"].startswith("test_col_")
        assert stats["total_chunks"] == 3
        assert stats["source_type_counts"]["email"] == 2
        assert stats["source_type_counts"]["document"] == 1

    def test_get_stats_empty(self, store):
        """빈 저장소 통계"""
        stats = store.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["source_type_counts"] == {}


# === 메타데이터 직렬화 라운드트립 테스트 ===


class TestMetadataRoundtrip:
    def test_list_metadata_roundtrip(self, store):
        """list 메타데이터가 저장/조회 시 유지"""
        chunks = [
            Chunk(
                content="테스트 내용입니다",
                metadata={
                    "participants": ["김감사", "박대리"],
                    "topics": ["감사", "비용"],
                    "count": 5,
                },
                source_type="email",
            )
        ]
        store.add_chunks(chunks)

        results = store.search("테스트", n_results=1, search_method="vector")
        assert len(results) == 1
        meta = results[0]["metadata"]
        assert meta["participants"] == ["김감사", "박대리"]
        assert meta["topics"] == ["감사", "비용"]
        assert meta["count"] == 5
