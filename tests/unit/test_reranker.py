"""Reranker 유닛 테스트 — FlagReranker를 모킹하여 동작 검증"""

from unittest.mock import MagicMock, patch

import pytest

from src.rag import reranker as reranker_module
from src.rag.reranker import Reranker, RerankerLoadError, get_reranker


def _sample_docs(n: int = 5) -> list[dict]:
    """vector_store.search 반환 형식의 샘플 문서"""
    return [
        {
            "content": f"문서 {i} 내용입니다.",
            "metadata": {"source_type": "email", "filename": f"doc_{i}.eml"},
            "score": 0.9 - i * 0.1,
            "search_method": "hybrid",
        }
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _reset_singleton():
    """테스트마다 싱글톤 초기화"""
    reranker_module._reranker_instance = None
    yield
    reranker_module._reranker_instance = None


class TestRerankerInit:
    def test_default_model(self):
        """기본 모델명은 settings에서"""
        r = Reranker()
        assert r.model_name == "BAAI/bge-reranker-v2-m3"
        assert r._model is None  # 지연 로딩

    def test_custom_model(self):
        """커스텀 모델명"""
        r = Reranker(model_name="custom/reranker")
        assert r.model_name == "custom/reranker"


class TestRerankerRerank:
    @patch("src.rag.reranker.Reranker._load_model")
    def test_empty_documents(self, mock_load):
        """빈 문서 리스트 → 빈 리스트 반환"""
        r = Reranker()
        result = r.rerank(query="질의", documents=[])
        assert result == []
        mock_load.assert_not_called()

    @patch("src.rag.reranker.Reranker._load_model")
    def test_rerank_scores_and_sorting(self, mock_load):
        """점수 기준 내림차순 정렬 + rerank_score 추가"""
        r = Reranker()
        r._model = MagicMock()
        # 역순 점수: doc0=0.1, doc1=0.5, doc2=0.9
        r._model.compute_score.return_value = [0.1, 0.5, 0.9]

        docs = _sample_docs(3)
        result = r.rerank(query="테스트", documents=docs, top_n=3)

        assert len(result) == 3
        # 0.9 > 0.5 > 0.1 순
        assert result[0]["rerank_score"] == pytest.approx(0.9)
        assert result[1]["rerank_score"] == pytest.approx(0.5)
        assert result[2]["rerank_score"] == pytest.approx(0.1)
        # 원본 content 보존
        assert result[0]["content"] == "문서 2 내용입니다."

    @patch("src.rag.reranker.Reranker._load_model")
    def test_rerank_top_n_truncation(self, mock_load):
        """top_n 초과 결과 잘림"""
        r = Reranker()
        r._model = MagicMock()
        r._model.compute_score.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]

        docs = _sample_docs(5)
        result = r.rerank(query="질의", documents=docs, top_n=2)

        assert len(result) == 2

    @patch("src.rag.reranker.Reranker._load_model")
    def test_rerank_single_document(self, mock_load):
        """단일 문서 — compute_score가 float 반환하는 경우"""
        r = Reranker()
        r._model = MagicMock()
        r._model.compute_score.return_value = 0.85  # float (not list)

        docs = _sample_docs(1)
        result = r.rerank(query="질의", documents=docs, top_n=1)

        assert len(result) == 1
        assert result[0]["rerank_score"] == pytest.approx(0.85)

    @patch("src.rag.reranker.Reranker._load_model")
    def test_rerank_preserves_original_fields(self, mock_load):
        """원본 dict 필드가 보존됨"""
        r = Reranker()
        r._model = MagicMock()
        r._model.compute_score.return_value = [0.7]

        docs = [{"content": "내용", "metadata": {"key": "val"}, "score": 0.5}]
        result = r.rerank(query="질의", documents=docs, top_n=1)

        assert result[0]["metadata"] == {"key": "val"}
        assert result[0]["score"] == 0.5  # 원본 score 유지
        assert result[0]["rerank_score"] == pytest.approx(0.7)

    @patch("src.rag.reranker.Reranker._load_model")
    def test_rerank_pairs_construction(self, mock_load):
        """query-document 쌍이 올바르게 구성되는지"""
        r = Reranker()
        r._model = MagicMock()
        r._model.compute_score.return_value = [0.5, 0.6]

        docs = [{"content": "A"}, {"content": "B"}]
        r.rerank(query="Q", documents=docs, top_n=2)

        r._model.compute_score.assert_called_once_with(
            [["Q", "A"], ["Q", "B"]], normalize=True
        )


class TestRerankerLoadError:
    def test_flagembedding_not_installed(self):
        """FlagEmbedding 미설치 시 명확한 에러"""
        r = Reranker()
        with patch.dict("sys.modules", {"FlagEmbedding": None}):
            with pytest.raises(RerankerLoadError, match="FlagEmbedding"):
                r._load_model()


class TestGetReranker:
    def test_singleton(self):
        """동일 모델이면 같은 인스턴스"""
        r1 = get_reranker()
        r2 = get_reranker()
        assert r1 is r2

    def test_different_model_new_instance(self):
        """다른 모델이면 새 인스턴스"""
        r1 = get_reranker()
        r2 = get_reranker(model_name="other/model")
        assert r1 is not r2
        assert r2.model_name == "other/model"
