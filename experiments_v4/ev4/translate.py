"""판정 참고용 번역 — 로컬 LLM(OLLAMA_LLM_MODEL, gemma4:12b) 전용, 캐시 (지시서 §6.3)

- 보안 모드 on 에서만 동작한다 (조사 텍스트 외부 전송 금지, 지시서 §2-1).
- 캐시 키 = (모델, 원문 해시). 같은 단위는 다시 번역하지 않는다.
- AST_FAKE_MODELS=1 이면 모델 없이 표시용 가짜 번역을 돌려준다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ev4.io import read_jsonl, write_jsonl
from src.utils.config import settings

NO_TRANSLATE_LANGS = ("ko", "en", "und")

_PROMPT = (
    "다음 텍스트를 한국어로 번역하세요. 내부 감사 판정 참고용이므로 의미를 빠짐없이 옮기고, "
    "인명·회사명·번호·금액·날짜는 원문 그대로 두세요. 번역문만 출력하세요.\n\n{text}"
)


class Translator:
    """로컬 LLM 번역기 (캐시 포함)

    Args:
        cache_path: 번역 캐시 JSONL
    """

    def __init__(self, cache_path: Path) -> None:
        if not settings.is_secure_mode:
            raise RuntimeError(
                "번역은 SECURITY_MODE=on 에서만 실행합니다 (조사 텍스트 외부 전송 금지)"
            )
        self.cache_path = cache_path
        self.model = "fake" if settings.ast_fake_models else settings.ollama_llm_model
        self._cache: dict[str, str] = {}
        if cache_path.exists():
            for r in read_jsonl(cache_path):
                self._cache[r["key"]] = r["ko"]
        self._llm = None
        self.n_new = 0

    def _key(self, text: str) -> str:
        return f"{self.model}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"

    def translate(self, text: str) -> str:
        """한국어 번역 (캐시 우선)"""
        if not text.strip():
            return ""
        key = self._key(text)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if settings.ast_fake_models:
            ko = f"[가짜 번역] {text[:200]}"
        else:
            if self._llm is None:
                from src.llm.router import LLMRouter

                self._llm = LLMRouter().get_llm(secure_mode=True)
            ko = str(self._llm.invoke([("human", _PROMPT.format(text=text))]).content).strip()
        self._cache[key] = ko
        self.n_new += 1
        return ko

    def save(self) -> None:
        """캐시 저장"""
        write_jsonl(self.cache_path, ({"key": k, "ko": v} for k, v in self._cache.items()))
