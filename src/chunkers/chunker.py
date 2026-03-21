"""스마트 청킹 엔진 — 문서용 / 채팅용 분리 처리"""

from dataclasses import dataclass, field
from datetime import datetime

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """청킹된 텍스트 단위"""
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""
    source_type: str = ""  # "email" | "teams_chat" | "document" | "attachment"


class DocumentChunker:
    """문서용 청커 — RecursiveCharacterTextSplitter 기반"""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size_docs
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap_docs

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """문서 텍스트를 청크로 분할"""
        # TODO: LangChain RecursiveCharacterTextSplitter 연동
        raise NotImplementedError


class ChatChunker:
    """채팅용 청커 — 시간 윈도우 기반 분할"""

    def __init__(self, window_minutes: int | None = None):
        self.window_minutes = window_minutes or settings.chat_window_minutes

    def chunk(self, messages: list[dict], metadata: dict | None = None) -> list[Chunk]:
        """채팅 메시지를 시간 윈도우 기준으로 대화 블록으로 분할

        Args:
            messages: [{"sender": str, "body": str, "timestamp": datetime}, ...]
            metadata: 공통 메타데이터 (participants, chat_type 등)
        """
        # TODO: 시간 윈도우 기반 청킹 구현
        raise NotImplementedError
