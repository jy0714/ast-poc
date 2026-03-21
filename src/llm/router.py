"""LLM 라우터 — 보안 모드에 따라 로컬/외부 LLM 분기"""

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMRouter:
    """보안 모드에 따라 적절한 LLM으로 라우팅

    - ON (secure):  Ollama gpt-oss:20b (로컬)
    - OFF (open):   OpenAI o3 (외부 API)
    """

    def __init__(self):
        self._ollama_llm = None
        self._openai_llm = None

    def get_llm(self, secure_mode: bool | None = None):
        """현재 보안 모드에 맞는 LLM 인스턴스 반환"""
        is_secure = secure_mode if secure_mode is not None else settings.is_secure_mode

        if is_secure:
            return self._get_ollama_llm()
        else:
            return self._get_openai_llm()

    def _get_ollama_llm(self):
        """Ollama 로컬 LLM 인스턴스"""
        # TODO: LangChain ChatOllama 초기화
        logger.info(f"로컬 LLM 사용: {settings.ollama_llm_model}")
        raise NotImplementedError

    def _get_openai_llm(self):
        """OpenAI 외부 LLM 인스턴스"""
        if not settings.openai_api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        # TODO: LangChain ChatOpenAI 초기화
        logger.info(f"외부 LLM 사용: {settings.openai_model}")
        raise NotImplementedError

    async def generate(
        self,
        prompt: str,
        context: list[str],
        secure_mode: bool | None = None,
    ) -> str:
        """RAG 프롬프트로 LLM 응답 생성"""
        # TODO: 컨텍스트 + 프롬프트 조합 → LLM 호출
        raise NotImplementedError

    async def generate_stream(
        self,
        prompt: str,
        context: list[str],
        secure_mode: bool | None = None,
    ):
        """RAG 프롬프트로 LLM 스트리밍 응답 생성"""
        # TODO: SSE 스트리밍 구현
        raise NotImplementedError
