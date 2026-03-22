"""LLM 라우터 — 보안 모드에 따라 로컬/외부 LLM 분기

보안 모드:
- ON (secure):  Ollama 로컬 LLM — 데이터 외부 전송 차단
- OFF (open):   OpenAI 외부 API — 검색된 청크만 외부 전송

RAG 프롬프트 템플릿을 사용하여 검색 결과 컨텍스트와
사용자 질의를 조합한 응답을 생성.
"""

from __future__ import annotations

from typing import AsyncIterator

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# RAG 시스템 프롬프트
_SYSTEM_PROMPT = """당신은 내부 감사(Audit) 지원 AI 어시스턴트입니다.
아래 제공된 검색 결과(Context)만을 근거로 사용자의 질문에 답변하세요.

규칙:
1. 검색 결과에 없는 내용은 추측하지 마세요. 근거가 부족하면 "제공된 자료에서 관련 내용을 찾을 수 없습니다"라고 답하세요.
2. 답변 시 어떤 자료에서 정보를 얻었는지 언급하세요 (예: "이메일에 따르면...", "첨부 문서에서...").
3. 한국어로 답변하세요.
4. 간결하고 명확하게 답변하세요."""

_USER_PROMPT_TEMPLATE = """## Context (검색 결과)

{context}

## 질문

{question}"""


class LLMRouter:
    """보안 모드에 따라 적절한 LLM으로 라우팅

    사용법:
        router = LLMRouter()
        answer = await router.generate("질문", ["컨텍스트1", "컨텍스트2"])
        async for token in router.generate_stream("질문", ["컨텍스트1"]):
            print(token, end="")
    """

    def __init__(self) -> None:
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
        """Ollama 로컬 LLM 인스턴스 (지연 생성)"""
        if self._ollama_llm is None:
            from langchain_ollama import ChatOllama

            self._ollama_llm = ChatOllama(
                model=settings.ollama_llm_model,
                base_url=settings.ollama_base_url,
                temperature=0.1,
            )
            logger.info(f"로컬 LLM 초기화: {settings.ollama_llm_model}")

        return self._ollama_llm

    def _get_openai_llm(self):
        """OpenAI 외부 LLM 인스턴스 (지연 생성)"""
        if self._openai_llm is None:
            if not settings.openai_api_key:
                raise ValueError(
                    "OpenAI API 키가 설정되지 않았습니다. "
                    ".env 파일에 OPENAI_API_KEY를 설정하세요."
                )

            from langchain_openai import ChatOpenAI

            self._openai_llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.1,
            )
            logger.info(f"외부 LLM 초기화: {settings.openai_model}")

        return self._openai_llm

    def _build_messages(
        self, question: str, context: list[str]
    ) -> list[tuple[str, str]]:
        """RAG 프롬프트 메시지 조합"""
        context_text = "\n\n---\n\n".join(
            f"[출처 {i + 1}]\n{c}" for i, c in enumerate(context)
        )
        if not context_text:
            context_text = "(검색 결과 없음)"

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            context=context_text,
            question=question,
        )

        return [
            ("system", _SYSTEM_PROMPT),
            ("human", user_prompt),
        ]

    async def generate(
        self,
        question: str,
        context: list[str],
        secure_mode: bool | None = None,
    ) -> str:
        """RAG 프롬프트로 LLM 응답 생성

        Args:
            question: 사용자 질문
            context: 검색된 청크 텍스트 리스트
            secure_mode: 보안 모드 (None이면 설정값 사용)

        Returns:
            LLM 응답 텍스트
        """
        llm = self.get_llm(secure_mode)
        messages = self._build_messages(question, context)

        try:
            response = await llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM 응답 생성 실패: {e}")
            raise ConnectionError(f"LLM 응답 생성 실패: {e}") from e

    async def generate_stream(
        self,
        question: str,
        context: list[str],
        secure_mode: bool | None = None,
    ) -> AsyncIterator[str]:
        """RAG 프롬프트로 LLM 스트리밍 응답 생성

        Args:
            question: 사용자 질문
            context: 검색된 청크 텍스트 리스트
            secure_mode: 보안 모드

        Yields:
            응답 토큰 문자열
        """
        llm = self.get_llm(secure_mode)
        messages = self._build_messages(question, context)

        try:
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"LLM 스트리밍 실패: {e}")
            raise ConnectionError(f"LLM 스트리밍 실패: {e}") from e
