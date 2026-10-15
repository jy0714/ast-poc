"""LLM 라우터 — 보안 모드에 따라 로컬/외부 LLM 분기

보안 모드:
- ON (secure):  Ollama 로컬 LLM — 데이터 외부 전송 차단
- OFF (open):   외부 API (OpenAI 또는 Anthropic) — 검색된 청크만 외부 전송
                settings.external_llm_provider로 provider 선택

RAG 프롬프트 템플릿을 사용하여 검색 결과 컨텍스트와
사용자 질의를 조합한 응답을 생성.
"""

from __future__ import annotations

import time
from typing import AsyncIterator, Callable

from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.token_counter import (
    build_usage_record,
    get_token_counter,
    log_token_usage,
    log_token_usage_detail,
)

logger = get_logger(__name__)

# RAG 시스템 프롬프트
_SYSTEM_PROMPT = """당신은 내부 감사(Audit) 지원 AI 어시스턴트입니다.
아래 제공된 검색 결과(Context)만을 **유일한 근거**로 사용하여 답변하세요.

## 절대 규칙

1. **Context 외 정보 사용 금지**: Context에 포함되지 않은 사실, 수치, 날짜, 이름은 절대 생성하지 마세요. 당신의 사전 학습 지식을 사용하지 마세요.
2. **출처 번호 인용 필수**: 모든 사실 진술에 반드시 `[출처 N]` 형식으로 해당 정보의 근거를 표기하세요. 출처 번호 없는 사실 진술은 금지됩니다.
   - 예: "비용이 15% 증가했습니다 [출처 2]."
   - 예: "김철수는 3월 회의에서 이를 언급했습니다 [출처 1][출처 3]."
3. **근거 부족 시 명시적 거절**: Context에서 답을 찾을 수 없으면 반드시 다음과 같이 답하세요:
   "제공된 자료에서 해당 내용을 찾을 수 없습니다."
   부분적으로만 답할 수 있는 경우, 답할 수 있는 부분만 출처와 함께 제시하고 나머지는 찾을 수 없다고 명시하세요.
4. **추론 vs 사실 구분**: Context에서 직접 확인되는 사실과 논리적 추론을 명확히 구분하세요. 추론 시 "~로 추정됩니다", "~가능성이 있습니다"로 표현하고, 근거 출처를 반드시 병기하세요.
5. **한국어 답변**: 한국어로 간결하고 명확하게 답변하세요.

## 답변 형식

- 핵심 내용을 먼저 제시하고, 각 문장마다 `[출처 N]`을 붙이세요.
- 여러 출처의 정보를 종합할 경우 해당 출처를 모두 병기하세요 (예: `[출처 1][출처 4]`).
- Context가 비어 있거나 "(검색 결과 없음)"이면 답변을 생성하지 말고 자료 부족을 알리세요."""

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
        self._anthropic_llm = None

    def get_llm(self, secure_mode: bool | None = None):
        """현재 보안 모드에 맞는 LLM 인스턴스 반환"""
        is_secure = secure_mode if secure_mode is not None else settings.is_secure_mode

        if is_secure:
            return self._get_ollama_llm()

        provider = (settings.external_llm_provider or "").lower()
        if provider == "anthropic":
            return self._get_anthropic_llm()
        if provider == "openai":
            return self._get_openai_llm()
        raise ValueError(
            f"지원하지 않는 external_llm_provider: '{settings.external_llm_provider}'. "
            f"'openai' 또는 'anthropic' 중 하나로 설정하세요."
        )

    def _get_ollama_llm(self):
        """Ollama 로컬 LLM 인스턴스 (지연 생성)"""
        if self._ollama_llm is None:
            from langchain_ollama import ChatOllama

            self._ollama_llm = ChatOllama(
                model=settings.ollama_llm_model,
                base_url=settings.ollama_base_url,
                temperature=settings.llm_temperature,
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
                temperature=settings.llm_temperature,
            )
            logger.info(f"외부 LLM 초기화 (OpenAI): {settings.openai_model}")

        return self._openai_llm

    def _get_anthropic_llm(self):
        """Anthropic Claude 외부 LLM 인스턴스 (지연 생성)"""
        if self._anthropic_llm is None:
            if not settings.anthropic_api_key:
                raise ValueError(
                    "Anthropic API 키가 설정되지 않았습니다. "
                    ".env 파일에 ANTHROPIC_API_KEY를 설정하세요."
                )

            from langchain_anthropic import ChatAnthropic

            self._anthropic_llm = ChatAnthropic(
                model=settings.anthropic_model,
                api_key=settings.anthropic_api_key,
                temperature=settings.llm_temperature,
            )
            logger.info(f"외부 LLM 초기화 (Anthropic): {settings.anthropic_model}")

        return self._anthropic_llm

    def _build_messages(
        self,
        question: str,
        context: list[str],
        extra_system_warning: str | None = None,
    ) -> list[tuple[str, str]]:
        """RAG 프롬프트 메시지 조합

        Args:
            extra_system_warning: 시스템 프롬프트 뒤에 덧붙일 추가 경고
                (관련성 낮은 검색 결과 등에 대한 방어용)
        """
        context_text = "\n\n---\n\n".join(
            f"[출처 {i + 1}]\n{c}" for i, c in enumerate(context)
        )
        if not context_text:
            context_text = "(검색 결과 없음)"

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            context=context_text,
            question=question,
        )

        system_prompt = _SYSTEM_PROMPT
        if extra_system_warning:
            system_prompt = f"{_SYSTEM_PROMPT}\n\n## 추가 주의\n\n{extra_system_warning}"

        return [
            ("system", system_prompt),
            ("human", user_prompt),
        ]

    async def generate(
        self,
        question: str,
        context: list[str],
        secure_mode: bool | None = None,
        extra_system_warning: str | None = None,
        case_id: str = "",
        on_complete: Callable[[dict], None] | None = None,
    ) -> str:
        """RAG 프롬프트로 LLM 응답 생성

        Args:
            question: 사용자 질문
            context: 검색된 청크 텍스트 리스트
            secure_mode: 보안 모드 (None이면 설정값 사용)
            extra_system_warning: 시스템 프롬프트에 삽입할 추가 경고 (선택)
            case_id: 토큰 로깅에 기록할 케이스 ID (선택)
            on_complete: 토큰 사용량 dict를 받는 완료 콜백 (선택)

        Returns:
            LLM 응답 텍스트
        """
        llm = self.get_llm(secure_mode)
        messages = self._build_messages(question, context, extra_system_warning)

        start = time.perf_counter()
        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            logger.error(f"LLM 응답 생성 실패: {e}")
            raise ConnectionError(f"LLM 응답 생성 실패: {e}") from e

        latency = time.perf_counter() - start
        self._record_token_usage(
            question=question,
            context=context,
            messages=messages,
            output_text=response.content or "",
            response=response,
            secure_mode=secure_mode,
            case_id=case_id,
            latency=latency,
            on_complete=on_complete,
        )
        return response.content

    async def generate_stream(
        self,
        question: str,
        context: list[str],
        secure_mode: bool | None = None,
        extra_system_warning: str | None = None,
        case_id: str = "",
        on_complete: Callable[[dict], None] | None = None,
    ) -> AsyncIterator[str]:
        """RAG 프롬프트로 LLM 스트리밍 응답 생성

        스트리밍 완료(또는 중단) 시 finally에서 토큰 사용량을 집계해 로깅하고
        on_complete 콜백을 호출한다. usage 정보는 마지막 chunk에 있으면 사용하고,
        없으면 누적된 출력 텍스트로 추정한다.

        Args:
            question: 사용자 질문
            context: 검색된 청크 텍스트 리스트
            secure_mode: 보안 모드
            extra_system_warning: 시스템 프롬프트에 삽입할 추가 경고 (선택)
            case_id: 토큰 로깅에 기록할 케이스 ID (선택)
            on_complete: 토큰 사용량 dict를 받는 완료 콜백 (선택)

        Yields:
            응답 토큰 문자열
        """
        llm = self.get_llm(secure_mode)
        messages = self._build_messages(question, context, extra_system_warning)

        start = time.perf_counter()
        collected: list[str] = []
        last_chunk = None
        try:
            async for chunk in llm.astream(messages):
                last_chunk = chunk
                if chunk.content:
                    collected.append(chunk.content)
                    yield chunk.content
        except Exception as e:
            logger.error(f"LLM 스트리밍 실패: {e}")
            raise ConnectionError(f"LLM 스트리밍 실패: {e}") from e
        finally:
            latency = time.perf_counter() - start
            self._record_token_usage(
                question=question,
                context=context,
                messages=messages,
                output_text="".join(collected),
                response=last_chunk,
                secure_mode=secure_mode,
                case_id=case_id,
                latency=latency,
                on_complete=on_complete,
            )

    def _record_token_usage(
        self,
        *,
        question: str,
        context: list[str],
        messages: list[tuple[str, str]],
        output_text: str,
        response,
        secure_mode: bool | None,
        case_id: str,
        latency: float,
        on_complete: Callable[[dict], None] | None,
    ) -> None:
        """토큰 사용량을 집계해 파일에 로깅하고 콜백을 호출 (실패해도 무해)"""
        try:
            is_secure = secure_mode if secure_mode is not None else settings.is_secure_mode
            if is_secure:
                model = settings.ollama_llm_model
            elif (settings.external_llm_provider or "").lower() == "anthropic":
                model = settings.anthropic_model
            else:
                model = settings.openai_model
            usage = get_token_counter().extract_usage_from_response(response)
            system_prompt = messages[0][1] if messages else ""

            tsv_record, detail_record = build_usage_record(
                question=question,
                context=context,
                output_text=output_text,
                model=model,
                secure_mode=is_secure,
                latency_sec=latency,
                usage=usage,
                system_prompt=system_prompt,
            )
            tsv_record["case_id"] = case_id
            detail_record["case_id"] = case_id

            log_token_usage(tsv_record)
            log_token_usage_detail(detail_record)

            logger.info(
                f"토큰 사용량: case={case_id or '-'}, model={model}, "
                f"in={tsv_record['input_tokens']}, out={tsv_record['output_tokens']}, "
                f"total={tsv_record['total_tokens']}, source={tsv_record['input_source']}, "
                f"latency={tsv_record['latency_sec']}s"
            )

            if on_complete is not None:
                on_complete(tsv_record)
        except Exception as e:
            logger.warning(f"토큰 사용량 수집 실패 (질의는 정상 처리됨): {e}")
