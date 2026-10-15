"""토크나이저(BM25용), 토큰 카운터(청크 크기·컨텍스트 비용용), 언어 판별(코퍼스 통계용)."""

from __future__ import annotations

import re
from typing import Callable, Iterable, Optional

_EDGE_PUNCT = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)

# 프로젝트 BM25 와 반드시 같은 태그셋을 써야 한다 (config 의 settings.kiwi_keep_tags 로 덮어쓰기).
DEFAULT_KIWI_KEEP_TAGS = (
    "NNG", "NNP", "NR", "VV", "VA", "XR", "SL", "SH", "SN",
    "W_SERIAL", "W_EMAIL", "W_URL", "W_HASHTAG", "W_MENTION",
)


def whitespace_tokenize(text: str) -> list[str]:
    """B2 베이스라인: 형태소 분석 없이 공백 분리 + 소문자 + 양끝 구두점 제거."""
    out = []
    for w in text.lower().split():
        w = _EDGE_PUNCT.sub("", w)
        if w:
            out.append(w)
    return out


class KiwiTokenizer:
    """kiwipiepy 형태소 분석 + POS 필터. 'VV-R'/'VV-I' 같은 세부 태그는 앞부분으로 비교한다."""

    def __init__(self, keep_tags: Optional[Iterable[str]] = None, num_workers: int = -1,
                 user_dict: Optional[str] = None, min_len: Optional[dict] = None):
        from kiwipiepy import Kiwi  # 지연 import: 공백 토크나이저만 쓸 때는 불필요

        # num_workers=-1: 전체 코어 사용 (일괄 분석은 멀티스레드 모드에서만 동작)
        self.kiwi = Kiwi(num_workers=num_workers)
        self.keep = set(keep_tags or DEFAULT_KIWI_KEEP_TAGS)
        # 최소 길이 (§3.4: 외국어·라틴 3자 이상, 그 외 2자 이상). None 이면 제한 없음
        ml = min_len or {}
        self.min_foreign = int(ml.get("foreign", 0) or 0)
        self.min_other = int(ml.get("other", 0) or 0)
        if user_dict:
            # 사내 은어·약어 사전 (kiwipiepy 형식: 단어<TAB>품사[<TAB>점수], 예: "법카\tNNP")
            self.kiwi.load_user_dictionary(str(user_dict))

    def _filter(self, tokens) -> list[str]:
        out = []
        for t in tokens:
            tag = t.tag if t.tag in self.keep else t.tag.split("-")[0]
            if tag not in self.keep:
                continue
            need = self.min_foreign if tag == "SL" else self.min_other
            if len(t.form) >= need:
                out.append(t.form.lower())
        return out

    def __call__(self, text: str) -> list[str]:
        return self._filter(self.kiwi.tokenize(text))

    def batch(self, texts: list[str]) -> list[list[str]]:
        try:
            return [self._filter(toks) for toks in self.kiwi.tokenize(texts)]
        except Exception:
            return [self(t) for t in texts]


def make_tokenizer(name: str, kiwi_keep_tags=None, kiwi_user_dict=None, kiwi_min_len=None) -> Callable[[str], list[str]]:
    """whitespace | kiwi (사용자 사전 적용) | kiwi_nodict (사용자 사전 미적용 — 사전 효과 절삭용)."""
    if name == "whitespace":
        return whitespace_tokenize
    if name == "kiwi":
        return KiwiTokenizer(kiwi_keep_tags, user_dict=kiwi_user_dict, min_len=kiwi_min_len)
    if name == "kiwi_nodict":
        return KiwiTokenizer(kiwi_keep_tags, min_len=kiwi_min_len)
    raise ValueError(f"알 수 없는 sparse_tokenizer: {name} (whitespace | kiwi | kiwi_nodict)")


def tokenize_many(tok, texts: list[str]) -> list[list[str]]:
    if hasattr(tok, "batch"):
        return tok.batch(texts)
    return [tok(t) for t in texts]


class TokenCounter:
    """'512 토큰' 의 기준 토크나이저. 기본은 XLM-R 계열(bge-m3 / bge-reranker-v2-m3 동일 vocab).

    로컬 HF 캐시 → 온라인 → 문자 수 근사 순으로 시도하고, 어떤 방식을 썼는지 `kind` 에 남긴다
    (논문 재현성 기술용). 사내망에서 HF 접근이 막혀 있으면 근사로 떨어지므로 run_meta 를 꼭 확인할 것.
    """

    def __init__(self, hf_name: Optional[str] = "BAAI/bge-reranker-v2-m3", chars_per_token: float = 2.0):
        self.tok = None
        self.cpt = max(0.5, float(chars_per_token))
        self.kind = f"approx:{self.cpt}chars/token"
        if hf_name:
            try:
                from transformers import AutoTokenizer  # type: ignore

                try:
                    self.tok = AutoTokenizer.from_pretrained(hf_name, local_files_only=True)
                except Exception:
                    self.tok = AutoTokenizer.from_pretrained(hf_name)
                self.kind = f"hf:{hf_name}"
            except Exception:
                self.tok = None
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        if self.tok is None:
            return max(1, round(len(text) / self.cpt)) if text else 0
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def count_cached(self, key: str, text: str) -> int:
        n = self._cache.get(key)
        if n is None:
            n = self._cache[key] = self.count(text)
        return n

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """토큰별 (start_char, end_char)."""
        if not text:
            return []
        if self.tok is None:
            step = max(1, round(self.cpt))
            return [(i, min(i + step, len(text))) for i in range(0, len(text), step)]
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        return [(s, e) for s, e in enc["offset_mapping"] if e > s]


_CZ = set("ěščřžůťďňĚŠČŘŽŮŤĎŇ")
_ES = set("ñÑ¿¡")
_ACC = set("áéíóúÁÉÍÓÚ")


def detect_lang(text: str) -> str:
    """가벼운 휴리스틱 (ko/zh/ja/cs/es/en/unknown). 정밀도가 필요하면 lingua 로 교체."""
    s = text[:3000]
    hangul = han = kana = latin = 0
    cz = es = acc = 0
    for c in s:
        o = ord(c)
        if 0xAC00 <= o <= 0xD7A3 or 0x3131 <= o <= 0x318E:
            hangul += 1
        elif 0x4E00 <= o <= 0x9FFF:
            han += 1
        elif 0x3040 <= o <= 0x30FF:
            kana += 1
        elif c.isalpha():
            latin += 1
            if c in _CZ:
                cz += 1
            elif c in _ES:
                es += 1
            elif c in _ACC:
                acc += 1
    letters = hangul + han + kana + latin
    if letters == 0:
        return "unknown"
    if hangul / letters >= 0.2:
        return "ko"
    if kana / letters >= 0.1:
        return "ja"
    if han / letters >= 0.3:
        return "zh"
    if cz >= 2:
        return "cs"
    if es >= 1 or acc >= 3:
        return "es"
    return "en"


def lang_bucket(text: str, ko_min: float = 0.7, en_max: float = 0.1) -> str:
    """증거 단위의 언어 구분 (논문 4.7절 언어별 Recall): ko | en | mixed | other | unknown.

    한글과 라틴 문자 중 한글 비율 r 로 판단한다. r ≥ ko_min → ko, r ≤ en_max 이고 라틴 문자 위주 → en,
    그 사이 → mixed. 한글·라틴 문자가 거의 없으면 detect_lang 결과(ja/zh 등)를 other 로 묶는다."""
    s = text[:3000]
    hangul = latin = 0
    for c in s:
        o = ord(c)
        if 0xAC00 <= o <= 0xD7A3 or 0x3131 <= o <= 0x318E:
            hangul += 1
        elif c.isascii() and c.isalpha():
            latin += 1
    if hangul + latin < 5:
        lang = detect_lang(s)
        return "unknown" if lang == "unknown" else ("other" if lang not in ("ko", "en") else lang)
    r = hangul / (hangul + latin)
    if r >= ko_min:
        return "ko"
    if r <= en_max:
        return "en"
    return "mixed"
