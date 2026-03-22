"""질의 파서 — 사용자 자연어 질의에서 의도·필터 추출

규칙 기반(정규식) 파서로, 사용자 질의에서:
1. 날짜 범위 추출  (예: "2025년 3월", "1월부터 3월까지")
2. 소스 타입 추출  (예: "이메일에서", "문서 중에")
3. 참여자 추출    (예: "홍길동이 보낸", "김철수 관련")
4. 검색 쿼리 정제  (필터 표현 제거 후 핵심 질의만 남김)

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

    # 4. 의도 분석
    intent = _detect_intent(original)

    # 5. 정제
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
