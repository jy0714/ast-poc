"""임베딩 서비스 — Ollama nomic-embed-text (항상 로컬)

보안 정책: 임베딩은 보안 모드와 무관하게 항상 로컬 Ollama에서 수행.
외부 API로 임베딩을 전송하지 않음.
"""

from __future__ import annotations

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Ollama 로컬 임베딩 모델 연동

    사용법:
        service = EmbeddingService()
        vector = service.embed_text("검색할 텍스트")
        vectors = service.embed_texts(["텍스트1", "텍스트2"])

        # LangChain 호환 객체 (ChromaDB 연동용)
        embeddings = service.get_langchain_embeddings()
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        """EmbeddingService 초기화

        Args:
            model: Ollama 임베딩 모델명 (기본: settings.ollama_embed_model)
            base_url: Ollama 서버 URL (기본: settings.ollama_base_url)
        """
        self.model = model or settings.ollama_embed_model
        self.base_url = base_url or settings.ollama_base_url
        self._langchain_embeddings = None

    def embed_text(self, text: str) -> list[float]:
        """단일 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (float 리스트)

        Raises:
            ConnectionError: Ollama 서버 연결 실패
        """
        embeddings = self.get_langchain_embeddings()
        try:
            return embeddings.embed_query(text)
        except Exception as e:
            raise ConnectionError(
                f"Ollama 임베딩 실패 ({self.base_url}, 모델: {self.model}): {e}"
            ) from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """배치 텍스트 임베딩

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트

        Raises:
            ConnectionError: Ollama 서버 연결 실패
        """
        if not texts:
            return []

        embeddings = self.get_langchain_embeddings()
        batch_size = settings.embed_batch_size
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                vectors = embeddings.embed_documents(batch)
                all_vectors.extend(vectors)
            except Exception as e:
                raise ConnectionError(
                    f"Ollama 배치 임베딩 실패 (batch {i // batch_size + 1}, "
                    f"{self.base_url}, 모델: {self.model}): {e}"
                ) from e

        logger.info(f"임베딩 완료: {len(texts)}개 텍스트")
        return all_vectors

    def get_langchain_embeddings(self):
        """LangChain 호환 임베딩 객체 반환

        Returns:
            OllamaEmbeddings 인스턴스 (LangChain 호환)
        """
        if self._langchain_embeddings is None:
            from langchain_ollama import OllamaEmbeddings

            self._langchain_embeddings = OllamaEmbeddings(
                model=self.model,
                base_url=self.base_url,
            )

        return self._langchain_embeddings
