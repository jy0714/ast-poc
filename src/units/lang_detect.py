"""단위별 언어 감지 (lingua-language-detector, 후보 언어 제한)

- 본문(헤더 제외)을 줄 단위로 본다. 한글·가나·한자는 문자 체계로 바로 세고, 라틴·키릴 구간만
  lingua 로 판정한다. 글자 토큰(숫자·코드 제외) 수로 가중한 최다 언어가 주 언어다.
- 토큰이 5개 미만이면 "und".
- 두 번째 언어의 토큰 비중이 20% 이상이면 혼용(lang_mixed=True, lang_second 기록).
- 분석(논문 언어별 결과)에만 쓰고 검색 경로를 나누는 데 쓰지 않는다.

v3.x 의 파일 단위 langdetect 는 한국어 메일 약 950건(12%)을 vi·et 로 오판정했다 (v4.1 V8).
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.text.bm25_tokenizer import tokenize
from src.units.email_clean import is_header_line

_HAS_DIGIT_RE = re.compile(r"\d")
_HAS_LETTER_RE = re.compile(r"[^\W\d_]")
# 문자 체계 (\u 이스케이프 — 리터럴로 쓰면 NFC 정규화로 범위가 바뀔 수 있음)
_HANGUL_RE = re.compile("[가-힣ᄀ-ᇿ㄰-㆏]+")
_CJK_RE = re.compile("[一-鿿㐀-䶿豈-﫿぀-ヿ]+")
_KANA_RE = re.compile("[぀-ヿ]")

# 주소·URL (mailto 포함) — 언어 판정 잡음
_NOISE_RE = re.compile(r"<?mailto:[^\s>]+>?|https?://\S+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# 후보 언어 (지시서 §5.1.8) — lingua Language 이름
CANDIDATE_LANGUAGES = (
    "KOREAN",
    "ENGLISH",
    "CHINESE",
    "JAPANESE",
    "GERMAN",
    "CZECH",
    "SPANISH",
    "FRENCH",
    "POLISH",
    "VIETNAMESE",
)
MIN_TOKENS = 5
MIXED_THRESHOLD = 0.20
UNDETERMINED = "und"


@dataclass(frozen=True)
class LangResult:
    """언어 감지 결과

    Attributes:
        lang: 주 언어 ISO 639-1 코드 ("ko", "en", "zh" …) 또는 "und"
        mixed: 두 번째 언어 비중이 20% 이상인지
        second: 두 번째 언어 코드 (혼용일 때만, 아니면 "")
        shares: 언어별 토큰 비중
    """

    lang: str
    mixed: bool
    second: str
    shares: dict[str, float]


_detector: Any = None
_detector_lock = threading.Lock()


def _get_detector() -> Any:
    """lingua LanguageDetector 싱글턴 (후보 언어 제한)"""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                from lingua import Language, LanguageDetectorBuilder

                langs = [getattr(Language, name) for name in CANDIDATE_LANGUAGES]
                _detector = LanguageDetectorBuilder.from_languages(*langs).build()
    return _detector


def _iso(language: Any) -> str:
    """lingua Language → ISO 639-1 소문자 코드"""
    return language.iso_code_639_1.name.lower()


def _word_weight(text: str) -> int:
    """언어 판정 가중치 = 글자로 된 토큰 수 (숫자·코드형 토큰은 제외)

    SAP 표처럼 코드(EX20231228043, G25N7-EB20)와 영어 열 이름이 많은 한국어 본문이
    코드 토큰 때문에 독일어 등으로 판정되던 문제를 막는다 (2단계 실측에서 발견).
    """
    return sum(
        1 for t in tokenize(text) if not _HAS_DIGIT_RE.search(t) and _HAS_LETTER_RE.search(t)
    )


def _line_votes(line: str, detector: Any) -> Counter[str]:
    """한 줄의 언어별 표(토큰 수)

    한글·가나·한자는 문자 체계만으로 언어가 정해지므로 직접 센다(한글→ko, 가나가 있으면
    한자·가나→ja, 아니면 한자→zh). 라틴·키릴 등 나머지 구간만 lingua 로 판정한다.
    줄 하나에 한 언어만 붙이면, 한국어 단어 몇 개가 라틴 코드 사이에 있을 때 줄 전체가
    스페인어 등으로 잡히는 문제가 있었다.
    """
    votes: Counter[str] = Counter()
    ko = _word_weight(" ".join(_HANGUL_RE.findall(line)))
    if ko:
        votes["ko"] += ko
    cjk = " ".join(_CJK_RE.findall(line))
    if cjk:
        votes["ja" if _KANA_RE.search(cjk) else "zh"] += len(tokenize(cjk))
    rest = _CJK_RE.sub(" ", _HANGUL_RE.sub(" ", line))
    w = _word_weight(rest)
    if w:
        lang = detector.detect_language_of(rest)
        if lang is not None:
            votes[_iso(lang)] += w
    return votes


def detect(text: str) -> LangResult:
    """본문 텍스트의 주 언어·혼용 여부 판정

    Args:
        text: 헤더를 제외한 본문

    Returns:
        LangResult
    """
    # 인용 헤더 줄(From:/To:/Cc: …)과 주소·URL은 본문 언어가 아니므로 판정에서 뺀다.
    # 전달 메일은 인용을 지우지 않아서, 로마자 이름이 줄줄이 나오는 수신자 줄이
    # 베트남어 등으로 오판정되어 주 언어를 뒤집는 문제가 있었다.
    lines = []
    for raw in (text or "").split("\n"):
        if not raw.strip() or is_header_line(raw):
            continue
        cleaned = _NOISE_RE.sub(" ", raw).strip()
        if cleaned:
            lines.append(cleaned)
    if sum(_word_weight(ln) for ln in lines) < MIN_TOKENS:
        return LangResult(UNDETERMINED, False, "", {})

    detector = _get_detector()
    totals: Counter[str] = Counter()
    for ln in lines:  # 줄마다 순차 판정 — 병렬 API는 워커 프로세스마다 스레드 풀을 띄움
        totals.update(_line_votes(ln, detector))
    total = sum(totals.values())
    if total < MIN_TOKENS:
        return LangResult(UNDETERMINED, False, "", {})

    ranked = totals.most_common()
    shares = {k: round(v / total, 4) for k, v in ranked}
    primary = ranked[0][0]
    second = ranked[1][0] if len(ranked) > 1 else ""
    mixed = bool(second) and shares[second] >= MIXED_THRESHOLD
    return LangResult(primary, mixed, second if mixed else "", shares)
