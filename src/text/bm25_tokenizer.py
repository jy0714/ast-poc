"""문자 체계별 BM25 토크나이저 (v4) — 색인과 질의에 반드시 같은 함수를 쓴다.

한 텍스트 안에 한국어·영어·체코어·중국어 등이 섞여도 언어 감지 없이 동작하도록,
텍스트를 문자 체계 구간으로 나누어 처리한다.

- 한글 구간      → kiwipiepy 형태소 (기본: 구두점·기호 태그만 제외)
- 한자·가나 구간 → 2글자 단위(bigram), 1글자 구간은 그대로
- 라틴·키릴 등   → 소문자 단어, 발음 구별 기호 제거(dvořák → dvorak)
- 코드형 문자열  → 통째 토큰 추가(INV-2024-0312 → inv-2024-0312). 분해 토큰은 위 규칙으로 함께 생성
- 금액           → 천 단위 구분 기호를 뺀 숫자 토큰 추가(1,200,000 → 1200000)
- 이메일 주소    → 통째 토큰 추가

v3.x `vector_store._kiwi_tokenize` 대비 해결하는 문제:
- 문장 끝 마침표가 붙은 라틴 단어가 잘림 (payment. → pa / yment.)
- 코드형 문자열의 숫자 부분이 W_SERIAL 태그라 품사 필터에서 버려짐 (INV-2024-0312 → inv)

규칙을 바꾸면 TOKENIZER_VERSION 을 올린다. 색인 메타의 버전과 다르면 재색인해야 한다.
(참조 구현: docs/v4/ref/script_tokenizer.py)
"""

from __future__ import annotations

import re
import threading
import unicodedata
from collections.abc import Iterable
from typing import Any

TOKENIZER_VERSION = "script-v1"

# 문자 범위는 \u 이스케이프로 쓴다. 리터럴로 쓰면 저장 시 NFC 정규화로 U+F900(호환 한자)이
# U+8C48 로 바뀌어 CJK 범위가 한글 음절(U+AC00~)까지 삼킨다.
_HANGUL = "가-힣ᄀ-ᇿ㄰-㆏"
_CJK = "一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ"

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)
_CODE_RE = re.compile(r"(?<![\w])[A-Za-z0-9]+(?:[-_/.][A-Za-z0-9]+)+(?![\w])")
_AMOUNT_RE = re.compile(r"(?<!\d)\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)")
_KO_RUN_RE = re.compile(f"[{_HANGUL}]+(?:[ \\t]+[{_HANGUL}]+)*")
_CJK_RUN_RE = re.compile(f"[{_CJK}]+")
_WORD_RE = re.compile(f"[^\\W_{_HANGUL}{_CJK}]+", re.UNICODE)

# kiwipiepy 품사 태그 중 항상 버리는 것: 문장부호·기호·특수문자
_KIWI_DROP_TAGS = frozenset({"SF", "SP", "SS", "SSO", "SSC", "SE", "SO", "SW", "SB"})
# ko_pos_filter=True 일 때만 쓰는 내용어 태그 (v3.x 방식 재현용)
_KIWI_CONTENT_TAGS = frozenset({"NNG", "NNP", "NNB", "VV", "VA", "MAG", "SN", "XR"})

_kiwi: Any = None
_kiwi_lock = threading.Lock()


def _get_kiwi() -> Any:
    """kiwipiepy.Kiwi 싱글턴 (지연 로드)"""
    global _kiwi
    if _kiwi is None:
        with _kiwi_lock:
            if _kiwi is None:
                from kiwipiepy import Kiwi

                _kiwi = Kiwi()
    return _kiwi


def _fold(word: str) -> str:
    """소문자 + 발음 구별 기호 제거"""
    word = unicodedata.normalize("NFKD", word.lower())
    return "".join(ch for ch in word if not unicodedata.combining(ch))


def _cjk_bigrams(run: str) -> list[str]:
    """한자·가나 구간 → 2글자 단위 (1글자 구간은 그대로)"""
    if len(run) == 1:
        return [run]
    return [run[i : i + 2] for i in range(len(run) - 1)]


def _korean_tokens(runs: list[str], ko_pos_filter: bool) -> list[str]:
    """한글 구간들을 Kiwi 형태소로 분석"""
    if not runs:
        return []
    kiwi = _get_kiwi()
    out: list[str] = []
    # 구간을 줄바꿈으로 이어 한 번만 호출(호출 오버헤드 감소). BM25 는 순서를 쓰지 않는다.
    for tok in kiwi.tokenize("\n".join(runs)):
        if tok.tag in _KIWI_DROP_TAGS:
            continue
        if ko_pos_filter and (tok.tag not in _KIWI_CONTENT_TAGS or len(tok.form) < 2):
            continue
        out.append(tok.form)
    return out


def tokenize(text: str, *, ko_pos_filter: bool = False) -> list[str]:
    """BM25 색인·질의 공용 토크나이저

    Args:
        text: 입력 텍스트 (헤더 포함 여부는 호출자가 결정)
        ko_pos_filter: True면 한글 구간에서 내용어 태그만 남김 (v3.x 재현용, 기본 False)

    Returns:
        토큰 리스트 (순서는 의미 없음, 중복 허용)
    """
    if not text:
        return []
    text = unicodedata.normalize("NFC", text)
    tokens: list[str] = []

    # 1) 통째로 보존할 토큰 (분해 토큰은 아래에서 별도로 생성됨)
    tokens.extend(m.group(0).lower() for m in _EMAIL_RE.finditer(text))
    tokens.extend(m.group(0).lower() for m in _CODE_RE.finditer(text))
    tokens.extend(m.group(0).replace(",", "") for m in _AMOUNT_RE.finditer(text))

    # 2) 한글 구간
    ko_runs = [m.group(0) for m in _KO_RUN_RE.finditer(text)]
    tokens.extend(_korean_tokens(ko_runs, ko_pos_filter))

    # 3) 한자·가나 구간
    for m in _CJK_RUN_RE.finditer(text):
        tokens.extend(_cjk_bigrams(m.group(0)))

    # 4) 그 밖의 문자 (라틴·키릴·그리스·숫자 등)
    tokens.extend(_fold(m.group(0)) for m in _WORD_RE.finditer(text))
    return tokens


def tokenize_many(texts: Iterable[str], *, ko_pos_filter: bool = False) -> list[list[str]]:
    """여러 텍스트를 같은 규칙으로 토큰화"""
    return [tokenize(t, ko_pos_filter=ko_pos_filter) for t in texts]
