"""RAG 질의 엔진 — 질의 파서 + 하이브리드 검색 + LLM 응답"""

from dataclasses import dataclass, field

from src.llm.router import LLMRouter
from src.vectorstore.vector_store import VectorStoreService
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """RAG 질의 결과"""
    answer: str
    sources: list[dict] = field(default_factory=list)
    filters_applied: dict = field(default_factory=dict)
    secure_mode: bool = True


class QueryParser:
    """사용자 질의에서 의도와 필터 조건 추출

    예시:
    - "나와 A씨가 가격 논의한 대화" → participants=["나", "A씨"], topic="가격"
    - "3월 14일 전후 B 프로젝트 관련 대화" → date_range=~3/14, topic="B 프로젝트"
    """

    def parse(self, query: str) -> dict:
        """질의 분석 → 검색 조건 추출

        Returns:
            {
                "search_query": str,        # 벡터 검색용 쿼리
                "filters": {                # 메타데이터 필터
                    "participants": [...],
                    "date_range": {...},
                    "source_type": str,
                },
            }
        """
        # TODO: LLM 기반 질의 파싱 또는 규칙 기반 파싱
        raise NotImplementedError


class RAGEngine:
    """RAG 질의 엔진 — 검색 + LLM 응답 생성 통합"""

    def __init__(self):
        self.query_parser = QueryParser()
        self.vector_store = VectorStoreService()
        self.llm_router = LLMRouter()

    async def query(self, user_query: str, secure_mode: bool = True) -> QueryResult:
        """사용자 질의 처리

        1. 질의 파싱 (의도 + 필터 추출)
        2. 하이브리드 검색 (벡터 + 메타데이터)
        3. 컨텍스트 조합
        4. LLM 응답 생성
        """
        # TODO: 전체 RAG 파이프라인 구현
        raise NotImplementedError

    async def query_stream(self, user_query: str, secure_mode: bool = True):
        """스트리밍 RAG 질의"""
        # TODO: 스트리밍 응답 구현
        raise NotImplementedError
