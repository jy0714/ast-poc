"""Reranker 유닛 테스트 — 백엔드 선택, rerank() 정렬·필터, bge 백엔드(FlagReranker 모킹)

Qwen3 백엔드의 입력 형식·절단·점수는 test_reranker_qwen3.py에서 검증한다.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.rag import reranker as reranker_module
from src.rag.reranker import (
    BgeRerankBackend,
    FakeRerankBackend,
    Qwen3RerankBackend,
    Reranker,
    RerankerLoadError,
    get_reranker,
    make_backend,
)
from src.utils.config import settings


def _sample_docs(n: int = 5) -> list[dict]:
    """검색 결과 형식의 샘플 문서"""
    return [
        {
            "content": f"문서 {i} 내용입니다.",
            "metadata": {"source_type": "email", "filename": f"doc_{i}.eml"},
            "score": 0.9 - i * 0.1,
            "search_method": "hybrid",
        }
        for i in range(n)
    ]


def _with_scores(scores: list[float]) -> Reranker:
    """백엔드 점수를 고정한 Reranker"""
    r = Reranker(backend="bge")
    r.backend.score = MagicMock(return_value=scores)
    return r


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    """테스트마다 싱글톤 초기화, 실제 모델 경로 사용 (가짜 모델 끔)"""
    monkeypatch.setattr(settings, "ast_fake_models", False)
    reranker_module._reranker_instance = None
    yield
    reranker_module._reranker_instance = None


class TestBackendSelection:
    def test_default_backend_is_qwen3(self, monkeypatch):
        """서비스 기본 백엔드는 qwen3"""
        monkeypatch.setattr(settings, "rerank_backend", "qwen3")
        monkeypatch.setattr(settings, "rerank_model", "Qwen/Qwen3-Reranker-4B")
        r = Reranker()
        assert isinstance(r.backend, Qwen3RerankBackend)
        assert r.model_name == "Qwen/Qwen3-Reranker-4B"
        assert r.backend.model is None  # 지연 로딩

    def test_bge_backend_uses_bge_model(self, monkeypatch):
        """bge 백엔드는 RERANK_BGE_MODEL을 쓴다"""
        monkeypatch.setattr(settings, "rerank_bge_model", "BAAI/bge-reranker-v2-m3")
        r = Reranker(backend="bge")
        assert isinstance(r.backend, BgeRerankBackend)
        assert r.model_name == "BAAI/bge-reranker-v2-m3"

    def test_same_max_length_both_backends(self, monkeypatch):
        """qwen3·bge가 같은 RERANK_MAX_LENGTH를 쓴다 (R9 공정성)"""
        monkeypatch.setattr(settings, "rerank_max_length", 4096)
        assert Qwen3RerankBackend().max_length == 4096
        assert BgeRerankBackend().max_length == 4096

    def test_fake_models_override(self, monkeypatch):
        """AST_FAKE_MODELS=1이면 백엔드 설정과 관계없이 가짜 재순위기"""
        monkeypatch.setattr(settings, "ast_fake_models", True)
        assert isinstance(make_backend("qwen3"), FakeRerankBackend)
        assert isinstance(make_backend("bge"), FakeRerankBackend)

    def test_unknown_backend(self):
        """알 수 없는 백엔드 이름 → ValueError"""
        with pytest.raises(ValueError):
            make_backend("colbert")


class TestRerankerRerank:
    def test_empty_documents(self):
        """빈 문서 리스트 → 빈 리스트, 백엔드 호출 없음"""
        r = _with_scores([])
        assert r.rerank(query="질의", documents=[]) == []
        r.backend.score.assert_not_called()

    def test_rerank_scores_and_sorting(self):
        """점수 기준 내림차순 정렬 + rerank_score 추가"""
        r = _with_scores([0.1, 0.5, 0.9])
        result = r.rerank(query="테스트", documents=_sample_docs(3), top_n=3, min_score=0.0)

        assert [d["rerank_score"] for d in result] == pytest.approx([0.9, 0.5, 0.1])
        assert result[0]["content"] == "문서 2 내용입니다."

    def test_ties_keep_input_order(self):
        """동점이면 입력(RRF) 순서를 유지한다"""
        r = _with_scores([0.5, 0.5, 0.5])
        result = r.rerank(query="q", documents=_sample_docs(3), top_n=3, min_score=0.0)
        assert [d["content"] for d in result] == [d["content"] for d in _sample_docs(3)]

    def test_rerank_top_n_truncation(self):
        """top_n 초과 결과 잘림"""
        r = _with_scores([0.9, 0.8, 0.7, 0.6, 0.5])
        assert len(r.rerank(query="질의", documents=_sample_docs(5), top_n=2)) == 2

    def test_min_score_filter(self):
        """min_score 미만 제거"""
        r = _with_scores([0.9, 0.05])
        result = r.rerank(query="q", documents=_sample_docs(2), top_n=5, min_score=0.1)
        assert len(result) == 1

    def test_rerank_preserves_original_fields(self):
        """원본 dict 필드가 보존됨"""
        r = _with_scores([0.7])
        docs = [{"content": "내용", "metadata": {"key": "val"}, "score": 0.5}]
        result = r.rerank(query="질의", documents=docs, top_n=1, min_score=0.0)

        assert result[0]["metadata"] == {"key": "val"}
        assert result[0]["score"] == 0.5
        assert result[0]["rerank_score"] == pytest.approx(0.7)

    def test_backend_receives_query_and_contents(self):
        """백엔드에 질의 원문과 content 목록이 그대로 전달된다"""
        r = _with_scores([0.5, 0.6])
        r.rerank(query="Q", documents=[{"content": "A"}, {"content": "B"}], top_n=2)
        r.backend.score.assert_called_once_with("Q", ["A", "B"])


class TestBgeBackend:
    def test_compute_score_uses_shared_max_length(self, monkeypatch):
        """FlagReranker 기본 512가 아니라 RERANK_MAX_LENGTH로 호출"""
        monkeypatch.setattr(settings, "rerank_max_length", 4096)
        b = BgeRerankBackend()
        b.model = MagicMock()
        b.model.tokenizer = None
        b.model.compute_score.return_value = [0.5, 0.6]

        assert b.score("Q", ["A", "B"]) == pytest.approx([0.5, 0.6])
        b.model.compute_score.assert_called_once_with(
            [["Q", "A"], ["Q", "B"]], normalize=True, max_length=4096
        )
        assert b.n_scored == 2

    def test_single_float_score(self):
        """compute_score가 float 하나를 돌려주는 경우"""
        b = BgeRerankBackend()
        b.model = MagicMock()
        b.model.tokenizer = None
        b.model.compute_score.return_value = 0.85
        assert b.score("q", ["d"]) == pytest.approx([0.85])

    def test_truncation_counted(self):
        """max_length를 넘는 쌍을 n_truncated에 센다"""
        b = BgeRerankBackend(max_length=5)
        b.model = MagicMock()
        b.model.tokenizer = MagicMock(
            return_value={"input_ids": [[1, 2, 3], [1, 2, 3, 4, 5, 6, 7]]}
        )
        b.model.compute_score.return_value = [0.1, 0.2]
        b.score("q", ["short", "long"])
        assert b.n_truncated == 1

    def test_flagembedding_not_installed(self):
        """FlagEmbedding 미설치 시 명확한 에러"""
        b = BgeRerankBackend()
        with (
            patch.dict("sys.modules", {"FlagEmbedding": None}),
            pytest.raises(RerankerLoadError, match="FlagEmbedding"),
        ):
            b._load()


class TestFakeBackend:
    def test_overlap_order(self):
        """질의 토큰이 많이 겹칠수록 높은 점수"""
        b = FakeRerankBackend()
        scores = b.score("payment approval", ["approval of payment", "payment only", "lunch"])
        assert scores[0] > scores[1] > scores[2]
        assert scores[2] == 0.0
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestGetReranker:
    def test_singleton(self):
        """같은 백엔드·모델이면 같은 인스턴스"""
        assert get_reranker(backend="bge") is get_reranker(backend="bge")

    def test_different_model_new_instance(self):
        """다른 모델이면 새 인스턴스"""
        r1 = get_reranker(backend="bge")
        r2 = get_reranker(backend="bge", model_name="other/model")
        assert r1 is not r2
        assert r2.model_name == "other/model"

    def test_different_backend_new_instance(self):
        """백엔드가 바뀌면 새 인스턴스"""
        r1 = get_reranker(backend="bge")
        r2 = get_reranker(backend="qwen3")
        assert r1 is not r2
        assert r2.backend_name == "qwen3"
