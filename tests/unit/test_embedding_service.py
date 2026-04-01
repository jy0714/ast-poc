"""EmbeddingService 유닛 테스트

Ollama 서버가 실행 중이어야 통과하는 테스트는 skip 처리.
"""

import pytest

from src.embeddings.embedding_service import (
    KNOWN_DIMENSIONS,
    EmbeddingDimensionError,
    EmbeddingService,
)


def _ollama_model_available() -> bool:
    """Ollama 서버 접근 + 현재 기본 임베딩 모델 존재 여부 확인"""
    try:
        import json
        import urllib.request

        from src.utils.config import settings

        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            model_names = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            return settings.ollama_embed_model in model_names
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_model_available(),
    reason="Ollama 서버가 실행되지 않거나 임베딩 모델이 설치되지 않았습니다",
)


class TestEmbeddingServiceInit:
    def test_default_settings(self):
        """기본 설정값으로 초기화"""
        service = EmbeddingService()
        assert service.model == "bge-m3"
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


class TestDimensionManagement:
    def test_known_model_dimension(self):
        """알려진 모델은 probe 없이 차원 반환"""
        service = EmbeddingService(model="bge-m3")
        assert service.get_dimension() == 1024

    def test_known_model_nomic(self):
        """nomic-embed-text 차원"""
        service = EmbeddingService(model="nomic-embed-text")
        assert service.get_dimension() == 768

    def test_known_model_with_tag(self):
        """태그가 포함된 모델명도 매칭"""
        service = EmbeddingService(model="bge-m3:latest")
        assert service.get_dimension() == 1024

    def test_dimension_cached(self):
        """차원이 캐싱됨"""
        service = EmbeddingService(model="bge-m3")
        _ = service.get_dimension()
        assert service._cached_dimension == 1024
        # 두 번째 호출은 캐시에서
        assert service.get_dimension() == 1024

    def test_validate_dimension_pass(self):
        """차원 일치 시 에러 없음"""
        service = EmbeddingService(model="bge-m3")
        service.validate_dimension(1024)  # 에러 없이 통과

    def test_validate_dimension_mismatch(self):
        """차원 불일치 시 EmbeddingDimensionError"""
        service = EmbeddingService(model="bge-m3")
        with pytest.raises(EmbeddingDimensionError, match="차원 불일치"):
            service.validate_dimension(768)

    def test_validate_dimension_error_message(self):
        """에러 메시지에 모델명과 차원 정보 포함"""
        service = EmbeddingService(model="nomic-embed-text")
        with pytest.raises(EmbeddingDimensionError) as exc_info:
            service.validate_dimension(1024)
        assert "nomic-embed-text" in str(exc_info.value)
        assert "768" in str(exc_info.value)
        assert "1024" in str(exc_info.value)

    def test_known_dimensions_registry(self):
        """KNOWN_DIMENSIONS에 주요 모델 등록"""
        assert "bge-m3" in KNOWN_DIMENSIONS
        assert "nomic-embed-text" in KNOWN_DIMENSIONS
        assert KNOWN_DIMENSIONS["bge-m3"] == 1024
        assert KNOWN_DIMENSIONS["nomic-embed-text"] == 768


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
