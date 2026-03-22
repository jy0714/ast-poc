"""EmbeddingService 유닛 테스트

Ollama 서버가 실행 중이어야 통과하는 테스트는 skip 처리.
"""

import pytest

from src.embeddings.embedding_service import EmbeddingService


def _ollama_available() -> bool:
    """Ollama 서버 접근 가능 여부 확인"""
    try:
        import urllib.request

        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama 서버가 실행되지 않고 있습니다",
)


class TestEmbeddingServiceInit:
    def test_default_settings(self):
        """기본 설정값으로 초기화"""
        service = EmbeddingService()
        assert service.model == "nomic-embed-text"
        assert "localhost" in service.base_url

    def test_custom_settings(self):
        """커스텀 설정값"""
        service = EmbeddingService(model="custom-model", base_url="http://custom:11434")
        assert service.model == "custom-model"
        assert service.base_url == "http://custom:11434"

    def test_get_langchain_embeddings_type(self):
        """LangChain 호환 객체 타입 검증"""
        service = EmbeddingService()
        embeddings = service.get_langchain_embeddings()
        from langchain_ollama import OllamaEmbeddings

        assert isinstance(embeddings, OllamaEmbeddings)

    def test_langchain_embeddings_cached(self):
        """LangChain 객체가 캐싱됨"""
        service = EmbeddingService()
        e1 = service.get_langchain_embeddings()
        e2 = service.get_langchain_embeddings()
        assert e1 is e2


@requires_ollama
class TestEmbeddingServiceWithOllama:
    def test_embed_text(self):
        """단일 텍스트 임베딩"""
        service = EmbeddingService()
        vector = service.embed_text("감사 보고서 검토")
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    def test_embed_texts_batch(self):
        """배치 임베딩"""
        service = EmbeddingService()
        texts = ["감사 보고서", "비용 분석", "회의록 검토"]
        vectors = service.embed_texts(texts)
        assert len(vectors) == 3
        assert all(len(v) > 0 for v in vectors)
        # 모든 벡터의 차원이 동일해야 함
        dims = {len(v) for v in vectors}
        assert len(dims) == 1

    def test_embed_texts_empty(self):
        """빈 리스트"""
        service = EmbeddingService()
        assert service.embed_texts([]) == []

    def test_embed_similar_texts(self):
        """유사한 텍스트는 유사한 벡터"""
        import math

        service = EmbeddingService()
        v1 = service.embed_text("감사 보고서 검토")
        v2 = service.embed_text("감사 보고서 리뷰")
        v3 = service.embed_text("오늘 점심 메뉴 추천")

        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na and nb else 0.0

        sim_similar = cosine_sim(v1, v2)
        sim_different = cosine_sim(v1, v3)
        # 유사한 텍스트의 코사인 유사도가 더 높아야 함
        assert sim_similar > sim_different


class TestEmbeddingServiceErrors:
    def test_connection_error(self):
        """연결 불가능한 서버 → ConnectionError"""
        service = EmbeddingService(base_url="http://localhost:19876")
        with pytest.raises(ConnectionError):
            service.embed_text("테스트")
