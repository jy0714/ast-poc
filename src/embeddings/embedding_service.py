"""임베딩 서비스 — Ollama nomic-embed-text (항상 로컬)"""

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Ollama 로컬 임베딩 모델 연동

    보안 정책: 임베딩은 보안 모드와 무관하게 항상 로컬에서 수행
    """

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.ollama_embed_model
        self.base_url = base_url or settings.ollama_base_url

    def embed_text(self, text: str) -> list[float]:
        """단일 텍스트 임베딩"""
        # TODO: Ollama 임베딩 API 호출
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """배치 텍스트 임베딩"""
        # TODO: 배치 처리 구현
        raise NotImplementedError

    def get_langchain_embeddings(self):
        """LangChain 호환 임베딩 객체 반환"""
        # TODO: OllamaEmbeddings 반환
        raise NotImplementedError
