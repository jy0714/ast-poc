"""토큰 카운터 — 파트 분할(UNIT_MAX_PART_TOKENS)과 단위 통계의 기준

v3.3 하네스 `TokenCounter` 이식 (docs/v4/ref/from_v33/text.py). 기준 토크나이저만
XLM-R → Qwen3(`Qwen/Qwen3-Embedding-4B`)로 바꿨다 — 주 모델(임베딩·재순위)이 같은
토크나이저를 쓰기 때문 (지시서 v4.1 V3).

로컬 HF 캐시 → 온라인 → 문자 수 근사 순으로 시도하고 어떤 방식을 썼는지 `kind`에 남긴다.
사내망에서 HF 접근이 막혀 있으면 근사로 떨어지므로 단위 통계(T1)에 `kind`를 함께 기록한다.
"""

from __future__ import annotations

import threading
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TokenCounter:
    """텍스트 토큰 수 계산기

    Args:
        hf_name: HF 토크나이저 이름. None 이면 바로 근사 모드
        chars_per_token: 근사 모드에서 문자 수 / 토큰 비율 (한국어 기준 보수적으로 2.0)
    """

    def __init__(
        self, hf_name: str | None = "Qwen/Qwen3-Embedding-4B", chars_per_token: float = 2.0
    ):
        self.tok: Any = None
        self.cpt = max(0.5, float(chars_per_token))
        self.kind = f"approx:{self.cpt}chars/token"
        if hf_name:
            try:
                from transformers import AutoTokenizer

                try:
                    self.tok = AutoTokenizer.from_pretrained(hf_name, local_files_only=True)
                except Exception:
                    self.tok = AutoTokenizer.from_pretrained(hf_name)
                self.kind = f"hf:{hf_name}"
            except Exception as e:
                logger.warning(f"토크나이저 로드 실패 → 문자 수 근사 사용 ({hf_name}): {e}")
                self.tok = None

    @property
    def is_exact(self) -> bool:
        """실제 토크나이저를 쓰는지 (근사가 아닌지)"""
        return self.tok is not None

    def count(self, text: str) -> int:
        """텍스트 하나의 토큰 수 (특수 토큰 제외)"""
        if not text:
            return 0
        if self.tok is None:
            return max(1, round(len(text) / self.cpt))
        return len(self.tok(text, add_special_tokens=False, verbose=False)["input_ids"])

    def count_many(self, texts: list[str]) -> list[int]:
        """여러 텍스트의 토큰 수 (배치 인코딩)"""
        if not texts:
            return []
        if self.tok is None:
            return [self.count(t) for t in texts]
        enc = self.tok(texts, add_special_tokens=False, verbose=False)["input_ids"]
        return [len(ids) if t else 0 for ids, t in zip(enc, texts, strict=True)]

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """토큰별 (start_char, end_char) — 한 줄이 상한을 넘을 때 강제 분할용"""
        if not text:
            return []
        if self.tok is None:
            step = max(1, round(self.cpt))
            return [(i, min(i + step, len(text))) for i in range(0, len(text), step)]
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        return [(s, e) for s, e in enc["offset_mapping"] if e > s]


_counter: TokenCounter | None = None
_counter_lock = threading.Lock()


def get_token_counter() -> TokenCounter:
    """설정(unit_token_hf_name) 기준 TokenCounter 싱글턴"""
    global _counter
    if _counter is None:
        with _counter_lock:
            if _counter is None:
                from src.utils.config import settings

                _counter = TokenCounter(settings.unit_token_hf_name)
    return _counter
