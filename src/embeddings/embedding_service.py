"""임베딩 서비스 — Ollama 로컬 임베딩 (항상 로컬)

보안 정책: 임베딩은 보안 모드와 무관하게 항상 로컬 Ollama에서 수행.
외부 API로 임베딩을 전송하지 않음.

모델별 벡터 차원:
- nomic-embed-text: 768
- bge-m3: 1024
- mxbai-embed-large: 1024

모델 변경 시 기존 컬렉션과 벡터 차원 불일치가 발생할 수 있으므로
validate_dimension()으로 호환성을 검증해야 함.
"""

from __future__ import annotations

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 모델별 알려진 벡터 차원 (Ollama 기준)
KNOWN_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 768,
    "bge-m3": 1024,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}


class EmbeddingDimensionError(Exception):
    """임베딩 벡터 차원 불일치 에러"""


class EmbeddingService:
    """Ollama 로컬 임베딩 모델 연동

    사용법:
        service = EmbeddingService()
        vector = service.embed_text("검색할 텍스트")
        vectors = service.embed_texts(["텍스트1", "텍스트2"])

        # 벡터 차원 확인
        dim = service.get_dimension()

        # 컬렉션 호환성 검증
        service.validate_dimension(existing_dim=768)
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
        self._cached_dimension: int | None = None

    def get_dimension(self) -> int:
        """현재 모델의 벡터 차원 반환

        알려진 모델이면 즉시 반환, 아니면 Ollama에 probe 요청.

        Returns:
            벡터 차원 수

        Raises:
            ConnectionError: Ollama 서버 연결 실패
        """
        if self._cached_dimension is not None:
            return self._cached_dimension

        # 알려진 모델이면 바로 반환
        if self.model in KNOWN_DIMENSIONS:
            self._cached_dimension = KNOWN_DIMENSIONS[self.model]
            return self._cached_dimension

        # 모델명에 알려진 키가 포함되어 있는지 확인 (tag 포함 모델명 대응)
        for known_model, dim in KNOWN_DIMENSIONS.items():
            if known_model in self.model:
                self._cached_dimension = dim
                return self._cached_dimension

        # 알 수 없는 모델 → probe
        logger.info(f"알 수 없는 모델 '{self.model}' — probe 임베딩으로 차원 확인")
        vector = self.embed_text("dimension probe")
        self._cached_dimension = len(vector)
        return self._cached_dimension

    def validate_dimension(self, existing_dim: int) -> None:
        """현재 모델의 벡터 차원이 기존 컬렉션과 호환되는지 검증

        Args:
            existing_dim: 기존 컬렉션의 벡터 차원

        Raises:
            EmbeddingDimensionError: 차원 불일치 시
        """
        current_dim = self.get_dimension()
        if current_dim != existing_dim:
            raise EmbeddingDimensionError(
                f"임베딩 차원 불일치: 현재 모델 '{self.model}'={current_dim}차원, "
                f"기존 컬렉션={existing_dim}차원. "
                f"모델 변경 후에는 기존 케이스를 재인덱싱해야 합니다."
            )

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
