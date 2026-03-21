"""문서 파서 — PDF, DOCX, PPTX, XLSX 텍스트 추출"""

from dataclasses import dataclass
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedDocument:
    """파싱된 문서"""
    filename: str
    content: str
    file_type: str  # "pdf" | "docx" | "pptx" | "xlsx"
    page_count: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentParser:
    """파일 타입에 따라 적절한 파서로 텍스트 추출"""

    SUPPORTED_TYPES = {".pdf", ".docx", ".pptx", ".xlsx", ".txt"}

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """파일 타입을 감지하고 적절한 파서로 텍스트 추출"""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED_TYPES:
            raise ValueError(f"지원하지 않는 파일 형식: {suffix}")

        logger.info(f"문서 파싱: {path.name} ({suffix})")

        parser_map = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".pptx": self._parse_pptx,
            ".xlsx": self._parse_xlsx,
            ".txt": self._parse_txt,
        }
        return parser_map[suffix](path)

    def parse_bytes(self, content: bytes, filename: str) -> ParsedDocument:
        """바이트 데이터에서 텍스트 추출 (첨부파일용)"""
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_TYPES:
            raise ValueError(f"지원하지 않는 파일 형식: {suffix}")

        # TODO: 임시 파일로 저장 후 파싱 또는 메모리에서 직접 파싱
        raise NotImplementedError("바이트 파싱은 다음 단계에서 구현됩니다.")

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        """PDF 텍스트 추출 (PyMuPDF)"""
        # TODO: PyMuPDF 구현
        raise NotImplementedError

    def _parse_docx(self, path: Path) -> ParsedDocument:
        """DOCX 텍스트 추출 (python-docx)"""
        # TODO: python-docx 구현
        raise NotImplementedError

    def _parse_pptx(self, path: Path) -> ParsedDocument:
        """PPTX 텍스트 추출 (python-pptx)"""
        # TODO: python-pptx 구현
        raise NotImplementedError

    def _parse_xlsx(self, path: Path) -> ParsedDocument:
        """XLSX 텍스트 추출 (openpyxl)"""
        # TODO: openpyxl 구현
        raise NotImplementedError

    def _parse_txt(self, path: Path) -> ParsedDocument:
        """TXT 텍스트 추출"""
        content = path.read_text(encoding="utf-8")
        return ParsedDocument(
            filename=path.name,
            content=content,
            file_type="txt",
        )
