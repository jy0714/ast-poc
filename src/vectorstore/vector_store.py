"""벡터 저장소 — ChromaDB 연동"""

from src.chunkers.chunker import Chunk
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreService:
    """ChromaDB 벡터 저장소 관리

    - 청크 저장 (벡터 + 메타데이터)
    - 하이브리드 검색 (벡터 유사도 + 메타데이터 필터)
    """

    def __init__(self, collection_name: str | None = None):
        self.collection_name = collection_name or settings.chroma_collection_name
        self._collection = None

    def _get_collection(self):
        """ChromaDB 컬렉션 가져오기/생성"""
        # TODO: ChromaDB 클라이언트 초기화 + 컬렉션 관리
        raise NotImplementedError

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """청크를 벡터 저장소에 추가

        Returns:
            추가된 청크 수
        """
        # TODO: 임베딩 생성 + ChromaDB 저장
        raise NotImplementedError

    def search(
        self,
        query: str,
        n_results: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """하이브리드 검색 (벡터 유사도 + 메타데이터 필터)

        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            filters: 메타데이터 필터 (participants, date_range, source_type 등)

        Returns:
            [{"content": str, "metadata": dict, "score": float}, ...]
        """
        # TODO: ChromaDB 검색 구현
        raise NotImplementedError

    def get_stats(self) -> dict:
        """저장소 통계 조회"""
        # TODO: 컬렉션 문서 수, 소스 타입별 분포 등
        raise NotImplementedError
