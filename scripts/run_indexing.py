"""인덱싱 파이프라인 — PST/문서 파싱 → 청킹 → 임베딩 → 벡터DB 저장"""

from pathlib import Path

from src.parsers.pst_parser import PSTParser
from src.parsers.document_parser import DocumentParser
from src.chunkers.chunker import DocumentChunker, ChatChunker
from src.vectorstore.vector_store import VectorStoreService
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IndexingPipeline:
    """전체 인덱싱 파이프라인 오케스트레이터"""

    def __init__(self):
        self.doc_parser = DocumentParser()
        self.doc_chunker = DocumentChunker()
        self.chat_chunker = ChatChunker()
        self.vector_store = VectorStoreService()

    def run(self) -> dict:
        """전체 인덱싱 실행

        1. data/input/pst/ 의 PST 파일 파싱
        2. data/input/documents/ 의 문서 파일 파싱
        3. 청킹
        4. 메타데이터 부착
        5. 임베딩 + 벡터DB 저장

        Returns:
            {"total_chunks": int, "pst_files": int, "doc_files": int, "errors": list}
        """
        # TODO: 파이프라인 구현
        raise NotImplementedError

    def index_pst(self, pst_path: Path) -> int:
        """단일 PST 파일 인덱싱"""
        # TODO: PST 파싱 → 분류 → 청킹 → 저장
        raise NotImplementedError

    def index_document(self, doc_path: Path) -> int:
        """단일 문서 파일 인덱싱"""
        # TODO: 문서 파싱 → 청킹 → 저장
        raise NotImplementedError


if __name__ == "__main__":
    pipeline = IndexingPipeline()
    result = pipeline.run()
    logger.info(f"인덱싱 완료: {result}")
