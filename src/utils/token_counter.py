"""토큰 사용량 계산 + 로깅 유틸리티

질의 1건당 입력/출력 토큰 수를 산출하고 별도 파일에 기록한다.
외부 API(Azure AI Foundry, OpenAI 등) 사용 시 비용 산정 근거 데이터로 사용.

토큰 수는 3가지 우선순위로 산출:
1. LLM 응답의 usage_metadata / response_metadata (있으면 가장 정확)
2. tiktoken (cl100k_base) — OpenAI 계열, 설치되어 있으면
3. 문자 기반 추정 (fallback) — 한·영 혼합 근사식

모든 파일 I/O는 try/except로 격리되어 로깅 실패가 질의 처리를 방해하지 않는다.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 동시 질의가 같은 파일에 줄을 쓸 때 줄 섞임 방지
_FILE_LOCK = threading.Lock()

# TSV 헤더 (스프레드시트로 바로 열 수 있게 탭 구분)
_TSV_COLUMNS = (
    "timestamp",
    "case_id",
    "question",
    "model",
    "security_mode",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_source",
    "context_chunks",
    "latency_sec",
)


class TokenCounter:
    """토큰 수 계산 유틸리티

    estimate_tokens로 텍스트 기반 추정을, extract_usage_from_response로
    LLM 응답 메타데이터에서 정확한 토큰 수를 추출한다.
    """

    def __init__(self) -> None:
        self._encoder: Any = None
        self._encoder_tried = False

    def _get_encoder(self) -> Any:
        """tiktoken cl100k_base 인코더 (지연 로드, 미설치 시 None)"""
        if not self._encoder_tried:
            self._encoder_tried = True
            try:
                import tiktoken

                self._encoder = tiktoken.get_encoding("cl100k_base")
                logger.debug("tiktoken cl100k_base 인코더 로드 완료")
            except Exception:
                # tiktoken 미설치 — 문자 기반 추정으로 fallback
                self._encoder = None
        return self._encoder

    def estimate_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 추정

        tiktoken이 있으면 정확 계산, 없으면 한·영 혼합 근사식 사용:
            max(utf-8 바이트 수 / 3, 문자 수 / 2)

        Args:
            text: 토큰 수를 셀 텍스트

        Returns:
            추정 토큰 수 (0 이상)
        """
        if not text:
            return 0

        encoder = self._get_encoder()
        if encoder is not None:
            try:
                return len(encoder.encode(text))
            except Exception:
                pass  # 인코딩 실패 시 추정으로 fallback

        byte_len = len(text.encode("utf-8"))
        return int(max(byte_len / 3, len(text) / 2))

    def extract_usage_from_response(self, response: Any) -> dict[str, Any] | None:
        """LLM 응답 메타데이터에서 토큰 사용량 추출

        - OpenAI (LangChain AIMessage): response.usage_metadata
          {'input_tokens', 'output_tokens', 'total_tokens'}
        - Ollama: response.response_metadata['prompt_eval_count' / 'eval_count']
        - OpenAI legacy: response.response_metadata['token_usage']

        Args:
            response: LangChain AIMessage 또는 유사 객체

        Returns:
            {'input_tokens', 'output_tokens', 'source'} 또는 None
        """
        if response is None:
            return None

        # 1. LangChain 표준 usage_metadata (OpenAI 등 외부 API)
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            inp = usage_meta.get("input_tokens")
            out = usage_meta.get("output_tokens")
            if inp is not None and out is not None:
                return {
                    "input_tokens": int(inp),
                    "output_tokens": int(out),
                    "source": "openai_api",
                }

        meta = getattr(response, "response_metadata", None) or {}

        # 2. Ollama — prompt_eval_count / eval_count
        prompt_eval = meta.get("prompt_eval_count")
        eval_count = meta.get("eval_count")
        if prompt_eval is not None or eval_count is not None:
            return {
                "input_tokens": int(prompt_eval or 0),
                "output_tokens": int(eval_count or 0),
                "source": "ollama_api",
            }

        # 3. OpenAI legacy — response_metadata['token_usage']
        token_usage = meta.get("token_usage") or {}
        if token_usage:
            return {
                "input_tokens": int(token_usage.get("prompt_tokens", 0)),
                "output_tokens": int(token_usage.get("completion_tokens", 0)),
                "source": "openai_api",
            }

        return None


_counter: TokenCounter | None = None


def get_token_counter() -> TokenCounter:
    """TokenCounter 싱글톤 (tiktoken 인코더 재사용)"""
    global _counter
    if _counter is None:
        _counter = TokenCounter()
    return _counter


def _resolve_path(path_str: str) -> Path:
    """상대 경로는 프로젝트 루트 기준으로 해석"""
    p = Path(path_str)
    if not p.is_absolute():
        p = settings.project_root / p
    return p


def _sanitize_question(question: str, max_len: int = 100) -> str:
    """질문을 TSV 한 칸에 안전하게 넣도록 정리 (탭/줄바꿈 → 공백, 길이 제한)"""
    cleaned = (question or "").replace("\t", " ").replace("\n", " ").replace("\r", " ")
    cleaned = cleaned.strip()
    return cleaned[:max_len]


def log_token_usage(record: dict[str, Any]) -> None:
    """질의 1건의 토큰 사용량을 TSV 파일에 1줄 append

    파일/디렉토리가 없으면 생성하고, 새 파일이면 헤더를 먼저 쓴다.
    실패해도 예외를 삼키고 WARNING 로그만 남긴다 (질의 처리에 영향 없음).

    Args:
        record: timestamp/case_id/question/model/security_mode/input_tokens/
            output_tokens/total_tokens/input_source/context_chunks/latency_sec 키
    """
    if not settings.token_log_enabled:
        return

    try:
        path = _resolve_path(settings.token_log_file)
        with _FILE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists() or path.stat().st_size == 0

            row = "\t".join(str(record.get(col, "")) for col in _TSV_COLUMNS)
            with path.open("a", encoding="utf-8") as f:
                if write_header:
                    f.write("\t".join(_TSV_COLUMNS) + "\n")
                f.write(row + "\n")
    except Exception as e:
        logger.warning(f"토큰 사용량 로깅 실패 (질의는 정상 처리됨): {e}")


def log_token_usage_detail(record: dict[str, Any]) -> None:
    """입력 토큰 상세 분해를 JSONL 파일에 1줄 append (옵션)

    token_log_detail=True일 때만 동작. 비용 분석 시 system/context/question별
    토큰 구성을 확인하는 용도.
    """
    if not settings.token_log_detail:
        return

    try:
        path = _resolve_path(settings.token_log_detail_file)
        with _FILE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"토큰 상세 로깅 실패 (질의는 정상 처리됨): {e}")


def build_usage_record(
    *,
    question: str,
    context: list[str],
    output_text: str,
    model: str,
    secure_mode: bool,
    latency_sec: float,
    usage: dict[str, Any] | None,
    system_prompt: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """토큰 사용량 TSV 레코드 + 상세 JSONL 레코드를 함께 산출

    LLM 응답 usage가 있으면(정확) 그 값을, 없으면 추정값을 사용한다.
    상세 레코드는 항상 추정 기반 분해(system/context/question)를 포함한다.

    Args:
        question: 사용자 질문 (LLM에 전달된 원본)
        context: LLM에 전달된 context 청크 리스트
        output_text: LLM 출력 전체 텍스트
        model: 사용된 모델명
        secure_mode: 보안 모드 (on/off 표기용)
        latency_sec: 응답 생성 소요 시간(초)
        usage: extract_usage_from_response 결과 (없으면 None)
        system_prompt: 실제 사용된 system 프롬프트 (상세 분해의 토큰 추정용)

    Returns:
        (tsv_record, detail_record)
    """
    counter = get_token_counter()

    # 추정 기반 입력 토큰 분해 (상세 로그용 — 항상 계산)
    system_tokens = counter.estimate_tokens(system_prompt)
    context_tokens = sum(counter.estimate_tokens(c) for c in context)
    question_tokens = counter.estimate_tokens(question)
    est_input = system_tokens + context_tokens + question_tokens
    est_output = counter.estimate_tokens(output_text)

    if usage:
        input_tokens = usage.get("input_tokens", est_input)
        output_tokens = usage.get("output_tokens", est_output)
        source = usage.get("source", "estimated")
    else:
        input_tokens = est_input
        output_tokens = est_output
        source = "estimated"

    total_tokens = input_tokens + output_tokens
    context_chunks = len(context)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tsv_record = {
        "timestamp": now,
        "case_id": "",  # 호출 측에서 채움
        "question": _sanitize_question(question),
        "model": model,
        "security_mode": "on" if secure_mode else "off",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_source": source,
        "context_chunks": context_chunks,
        "latency_sec": round(latency_sec, 3),
    }

    detail_record = {
        "timestamp": now,
        "case_id": "",
        "question": _sanitize_question(question),
        "model": model,
        "breakdown": {
            "system_prompt_tokens": system_tokens,
            "context_tokens": context_tokens,
            "question_tokens": question_tokens,
            "total_input_tokens": est_input,
        },
        "context_chunks": context_chunks,
        "avg_chunk_tokens": round(context_tokens / context_chunks, 1) if context_chunks else 0,
        "output_tokens": output_tokens,
        "source": source,
    }

    return tsv_record, detail_record
