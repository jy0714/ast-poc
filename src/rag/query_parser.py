"""질의 파서 — 사용자 자연어 질의에서 의도·필터 추출

규칙 기반(정규식) 파서로, 사용자 질의에서:
1. 날짜 범위 추출  (예: "2025년 3월", "1월부터 3월까지")
2. 소스 타입 추출  (예: "이메일에서", "문서 중에")
3. 참여자 추출    (예: "홍길동이 보낸", "김철수 관련")
4. 작성자/수정자 추출 (예: "홍길동이 만든 견적서", "alice가 수정한 문서",
                       "작성자: 김철수", "modified by bob")
   → 메타필터(author, last_modified_by)로 변환되어 ChromaDB 검색에 적용됨.
   ChromaDB는 정확 매칭이므로 메타데이터의 author와 정확히 일치해야 함
   (Office/PDF는 일반적으로 사용자 이름 그대로 저장됨).
5. 검색 쿼리 정제  (필터 표현 제거 후 핵심 질의만 남김)

향후 LLM 기반 의도 분석으로 교체/보강 가능.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedQuery:
    """파싱된 질의 결과"""

    original: str  # 원본 질의
    cleaned: str  # 필터 표현 제거 후 정제된 검색 쿼리
    filters: dict[str, Any] = field(default_factory=dict)
    intent: str = "search"  # "search" | "summarize" | "compare" | "timeline"


# === 소스 타입 매핑 ===

_SOURCE_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"이메일|메일|mail|email|eml", re.IGNORECASE), "email"),
    (re.compile(r"팀즈|teams|채팅|chat|대화", re.IGNORECASE), "teams_chat"),
    (re.compile(r"문서|보고서|리포트|document|report|pdf|docx|pptx|xlsx", re.IGNORECASE), "document"),
    (re.compile(r"첨부\s*파일|attachment", re.IGNORECASE), "attachment"),
]

# 소스 타입 필터 표현 (질의에서 제거할 패턴)
_SOURCE_FILTER_EXPR = re.compile(
    r"(이메일|메일|팀즈|teams|채팅|문서|보고서|첨부\s*파일)"
    r"\s*(에서|중에서?|내에서?|으로|만|에서만|중에|속에서?)\s*",
    re.IGNORECASE,
)

# === 날짜 패턴 ===

# "2025년 3월" → year-month
_DATE_YM = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월")
# "2025년" → full year
_DATE_Y = re.compile(r"(\d{4})\s*년")
# "3월" → month only (current/recent year assumed)
_DATE_M = re.compile(r"(\d{1,2})\s*월")
# "1월부터 3월까지" or "1월~3월"
_DATE_RANGE = re.compile(
    r"(\d{1,2})\s*월\s*(?:부터|에서)\s*(\d{1,2})\s*월\s*(?:까지|사이)",
)
# "2025-01-01" ~ "2025-03-31" ISO format
_DATE_ISO = re.compile(r"(\d{4}-\d{2}-\d{2})")
# 날짜 필터 표현 (질의에서 제거)
_DATE_FILTER_EXPR = re.compile(
    r"\d{4}\s*년\s*\d{1,2}\s*월\s*(?:부터|에서)?\s*"
    r"(?:\d{1,2}\s*월\s*(?:까지|사이))?\s*"
    r"|(?:최근|지난)\s*\d+\s*(?:개월|달|주|일)\s*(?:간|동안|내)?\s*"
    r"|\d{4}\s*년\s*"
    r"|\d{1,2}\s*월\s*(?:에|의|부터|까지)?\s*",
)

# === 참여자 패턴 ===

# "홍길동이 보낸", "김철수에게 온", "박영희 관련"
# 조사(이/가/에게 등)를 캡처 그룹 밖에 두어 이름만 추출
_PARTICIPANT_EXPR = re.compile(
    r"([가-힣]{2,4})(?:이|가|에게|한테|께서|으?로부터)"
    r"\s*(?:보낸|받은|작성|참여|관련|온|쓴|메일|이메일)",
)

# === 작성자/수정자 패턴 ===

# 이름 매칭 — 한글 2~4자, 영문 단어, 이메일 주소를 모두 허용.
# 영문은 단어 경계로 안전하게 매칭하기 위해 \b로 감쌈 (사용처에서 부착).
_NAME_PATTERN = r"(?:[가-힣]{2,4}|[A-Za-z][A-Za-z0-9._\-]*(?:@[A-Za-z0-9._\-]+)?)"

# 동사 뒤에 한글이 더 붙으면(예: "수정자", "작성자") 동사로 안 잡도록 negative lookahead.
# 영문도 단어 경계로 안전하게 분리.

# 작성자 — 한국어 "이름이/가 만든/작성한/생성한/쓴/기안한"
# 연결형(만들고/작성하고/생성하고)도 포함하여 "alice가 만들고 bob이 수정한"
# 같은 복합 표현에서 author도 추출되도록 함.
_AUTHOR_BY_VERB_KO = re.compile(
    rf"({_NAME_PATTERN})\s*(?:이|가|이가)?\s*"
    r"(?:만들었던|만들고|만든|"
    r"작성했던|작성하고|작성한|"
    r"생성하고|생성한|쓰고|쓴|기안하고|기안한)(?![가-힣])"
)
# 작성자 — 영어 "created/written/authored/made by 이름"
_AUTHOR_BY_VERB_EN = re.compile(
    rf"\b(?:created|written|authored|made|drafted)\s+by\s+({_NAME_PATTERN})\b",
    re.IGNORECASE,
)
# 작성자 — 명시 라벨 "작성자: 이름", "author: 이름"
# 영문 'author'는 단어 경계 강제 (예: "authored" 내부 매칭 방지)
_AUTHOR_LABEL = re.compile(
    rf"(?:\bauthor\b|작성자)\s*(?:은|는|이|가|:|=)?\s+({_NAME_PATTERN})",
    re.IGNORECASE,
)

# 수정자 — 한국어 "이름이/가 (마지막으로) 수정한/편집한/업데이트한"
_MODIFIER_BY_VERB_KO = re.compile(
    rf"({_NAME_PATTERN})\s*(?:이|가|이가)?\s*"
    r"(?:마지막으?로\s*)?"
    r"(?:수정하고|수정한|편집하고|편집한|업데이트하고|업데이트한)(?![가-힣])"
)
# 수정자 — 영어 "(last) modified/edited/revised/updated by 이름"
_MODIFIER_BY_VERB_EN = re.compile(
    rf"\b(?:last\s+)?(?:modified|edited|revised|updated)\s+by\s+({_NAME_PATTERN})\b",
    re.IGNORECASE,
)
# 수정자 — 명시 라벨 "수정자: 이름", "last modified by: 이름"
_MODIFIER_LABEL = re.compile(
    rf"(?:마지막\s*)?(?:\blast\s+modifier\b|수정자|편집자)"
    rf"\s*(?:은|는|이|가|:|=)?\s+({_NAME_PATTERN})",
    re.IGNORECASE,
)

# === 의도 패턴 ===

_INTENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"시간\s*순|타임라인|timeline|연대기|경과", re.IGNORECASE), "timeline"),
    (re.compile(r"비교|차이|compare|diff", re.IGNORECASE), "compare"),
    (re.compile(r"요약|정리|summarize|summary", re.IGNORECASE), "summarize"),
]


def parse_query(query: str, year_hint: int = 2025) -> ParsedQuery:
    """사용자 질의를 파싱하여 필터와 정제된 쿼리를 추출

    Args:
        query: 사용자 원본 질의
        year_hint: 월만 지정된 경우 사용할 연도 (기본: 2025)

    Returns:
        ParsedQuery (정제된 쿼리 + 추출된 필터 + 의도)
    """
    if not query or not query.strip():
        return ParsedQuery(original=query, cleaned=query)

    original = query.strip()
    filters: dict[str, Any] = {}
    cleaned = original

    # 1. 소스 타입 추출
    source_types = _extract_source_types(original)
    if source_types:
        filters["source_type"] = source_types[0] if len(source_types) == 1 else source_types
        cleaned = _SOURCE_FILTER_EXPR.sub(" ", cleaned)

    # 2. 날짜 범위 추출
    date_filters = _extract_date_filters(original, year_hint)
    if date_filters:
        filters.update(date_filters)
        cleaned = _DATE_FILTER_EXPR.sub(" ", cleaned)

    # 3. 참여자 추출
    participants = _extract_participants(original)
    if participants:
        filters["participants"] = participants
        # 참여자 이름은 검색 쿼리에 남겨둠 (검색에도 유용)

    # 4. 작성자/수정자 추출 (ChromaDB 메타필터로 직결)
    authors = _extract_authors(original)
    if authors:
        filters["author"] = authors[0] if len(authors) == 1 else authors
    modifiers = _extract_modifiers(original)
    if modifiers:
        filters["last_modified_by"] = (
            modifiers[0] if len(modifiers) == 1 else modifiers
        )

    # 5. 의도 분석
    intent = _detect_intent(original)

    # 6. 정제
    cleaned = _clean_query(cleaned)

    # 정제 후 빈 쿼리면 원본 사용
    if not cleaned:
        cleaned = original

    return ParsedQuery(
        original=original,
        cleaned=cleaned,
        filters=filters,
        intent=intent,
    )


def _extract_source_types(text: str) -> list[str]:
    """소스 타입 키워드 추출"""
    found: list[str] = []
    for pattern, source_type in _SOURCE_TYPE_PATTERNS:
        if pattern.search(text):
            if source_type not in found:
                found.append(source_type)
    return found


def _extract_date_filters(text: str, year_hint: int) -> dict[str, str]:
    """날짜 범위 필터 추출"""
    filters: dict[str, str] = {}

    # ISO 날짜 (최우선)
    iso_dates = _DATE_ISO.findall(text)
    if len(iso_dates) >= 2:
        filters["date_range_start"] = iso_dates[0]
        filters["date_range_end"] = iso_dates[1]
        return filters
    if len(iso_dates) == 1:
        filters["date"] = iso_dates[0]
        return filters

    # "1월부터 3월까지" 범위
    range_match = _DATE_RANGE.search(text)
    if range_match:
        m_start, m_end = int(range_match.group(1)), int(range_match.group(2))
        # 연도 찾기
        ym = _DATE_YM.search(text)
        year = int(ym.group(1)) if ym else year_hint
        filters["date_range_start"] = f"{year}-{m_start:02d}-01"
        # 월 말일 계산
        if m_end == 12:
            filters["date_range_end"] = f"{year}-12-31"
        else:
            filters["date_range_end"] = f"{year}-{m_end + 1:02d}-01"
        return filters

    # "2025년 3월" 특정 월
    ym = _DATE_YM.search(text)
    if ym:
        year, month = int(ym.group(1)), int(ym.group(2))
        filters["date_range_start"] = f"{year}-{month:02d}-01"
        if month == 12:
            filters["date_range_end"] = f"{year}-12-31"
        else:
            filters["date_range_end"] = f"{year}-{month + 1:02d}-01"
        return filters

    # "2025년" 전체 연도
    y = _DATE_Y.search(text)
    if y:
        year = int(y.group(1))
        filters["date_range_start"] = f"{year}-01-01"
        filters["date_range_end"] = f"{year}-12-31"
        return filters

    return filters


def _extract_participants(text: str) -> list[str]:
    """참여자 이름 추출"""
    matches = _PARTICIPANT_EXPR.findall(text)
    # 중복 제거, 순서 유지
    seen: set[str] = set()
    result: list[str] = []
    for name in matches:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _extract_authors(text: str) -> list[str]:
    """작성자(author) 추출 — 한/영 동사 + 라벨 패턴 통합

    여러 패턴이 같은 이름을 잡으면 중복 제거. "alice가 만들고 bob이 작성한"
    같은 복합 표현은 여러 author로 추출 가능.
    """
    matches: list[str] = []
    for pattern in (_AUTHOR_BY_VERB_KO, _AUTHOR_BY_VERB_EN, _AUTHOR_LABEL):
        matches.extend(pattern.findall(text))
    cleaned = [_strip_korean_particle(n) for n in matches]
    return _dedupe_preserve_order(cleaned)


def _extract_modifiers(text: str) -> list[str]:
    """수정자(last_modified_by) 추출 — 한/영 동사 + 라벨 패턴 통합"""
    matches: list[str] = []
    for pattern in (_MODIFIER_BY_VERB_KO, _MODIFIER_BY_VERB_EN, _MODIFIER_LABEL):
        matches.extend(pattern.findall(text))
    cleaned = [_strip_korean_particle(n) for n in matches]
    return _dedupe_preserve_order(cleaned)


def _strip_korean_particle(name: str) -> str:
    """한글 이름 뒤에 붙은 격조사(이/가/이가)를 제거

    regex가 한글 이름의 그리디 매칭으로 "홍길동이"처럼 조사를 함께 캡처하는
    경우가 있어 후처리로 분리. 영문/이메일은 영향 없음.

    한계: "정민이"처럼 격조사 없이 "이"로 끝나는 이름은 1글자 손실됨.
    PoC에서 받아들이는 trade-off (Office 메타데이터의 author는 보통 회사
    사용자 이름이거나 영어/이메일 형식이라 빈도 낮음).
    """
    if not re.search(r"[가-힣]", name):
        return name  # 영문/이메일은 그대로
    if name.endswith("이가"):
        return name[:-2]
    if name.endswith(("이", "가")) and len(name) >= 3:
        # 2글자 이름은 조사 떼면 1글자가 되어 의미 없으므로 보존
        return name[:-1]
    return name


def _dedupe_preserve_order(names: list[str]) -> list[str]:
    """순서를 유지하면서 중복 제거 (대소문자 무시)"""
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        key = name.lower().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(name.strip())
    return result


def _detect_intent(text: str) -> str:
    """질의 의도 분석"""
    for pattern, intent in _INTENT_PATTERNS:
        if pattern.search(text):
            return intent
    return "search"


def _clean_query(text: str) -> str:
    """필터 표현 제거 후 쿼리 정제"""
    # 연속 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    # 앞뒤 조사/접속사 제거
    text = re.sub(r"^[에서의를은는이가과와]+\s*", "", text)
    text = re.sub(r"\s+[에서의를은는이가과와]+$", "", text)
    return text.strip()
