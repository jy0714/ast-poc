"""유형별 증거 단위 헤더 (레이블은 본문 언어와 관계없이 한국어로 고정)

    [이메일] 2024-03-05 14:22 | 보낸이: 홍길동 | 받는이: 김철수, 이영희 | 참조: 박민수 | 제목: RE: 3월 정산 건 | 첨부: 견적서.xlsx
    [채팅] 2024-03-05 14:00~14:47 | 참여자: 홍길동, 김철수 | 방: 정산팀
    [문서] 정산기준_v3.docx | 3/12쪽 | 작성자: 홍길동 | 수정일: 2024-02-28
    [첨부] 견적서.xlsx | 2/3쪽 | 부모 메일: 2024-03-05 홍길동→김철수 "RE: 3월 정산 건"

- 빈 필드는 생략한다.
- 받는이·참여자·첨부가 5개를 넘으면 앞 5개 뒤에 "외 n명"(첨부는 "외 n건")을 붙인다.
- 시각은 settings.unit_header_tz(기본 Asia/Seoul)로 바꿔 표시한다. 메타데이터 date 는 원본 그대로.
- 헤더 형식은 체크포인트 2에서 사용자가 확정한다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

MAX_NAMES = 5

# page_kind → 위치 표기
_POSITION_FORMATS = {
    "page": "{n}/{total}쪽",
    "slide": "슬라이드 {n}/{total}",
    "block": "{n}/{total}부",
}


def _tz() -> ZoneInfo | None:
    from src.utils.config import settings

    try:
        return ZoneInfo(settings.unit_header_tz) if settings.unit_header_tz else None
    except Exception:
        return None


def parse_dt(value: Any) -> datetime | None:
    """datetime 또는 ISO 8601 문자열 → datetime (실패 시 None)"""
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _local(dt: datetime) -> datetime:
    """시간대 정보가 있으면 헤더 표시 시간대로 변환 (naive 는 그대로)"""
    tz = _tz()
    if dt.tzinfo is not None and tz is not None:
        return dt.astimezone(tz)
    return dt


def fmt_datetime(value: Any) -> str:
    """'YYYY-MM-DD HH:MM' (표시 시간대 기준). 해석 불가면 원문 앞 16자"""
    dt = parse_dt(value)
    if dt is None:
        return str(value or "")[:16]
    return _local(dt).strftime("%Y-%m-%d %H:%M")


def fmt_date(value: Any) -> str:
    """'YYYY-MM-DD' (표시 시간대 기준). 해석 불가면 원문 앞 10자"""
    dt = parse_dt(value)
    if dt is None:
        return str(value or "")[:10]
    return _local(dt).strftime("%Y-%m-%d")


def fmt_range(start: Any, end: Any) -> str:
    """채팅 세션 시간 범위: 같은 날이면 'YYYY-MM-DD HH:MM~HH:MM'"""
    s, e = parse_dt(start), parse_dt(end)
    if s is None:
        return ""
    s = _local(s)
    if e is None:
        return s.strftime("%Y-%m-%d %H:%M")
    e = _local(e)
    if s.date() == e.date():
        return f"{s:%Y-%m-%d %H:%M}~{e:%H:%M}"
    return f"{s:%Y-%m-%d %H:%M}~{e:%Y-%m-%d %H:%M}"


def join_limited(items: list[str], unit: str = "명", limit: int = MAX_NAMES) -> str:
    """앞 limit 개만 쉼표로 잇고 나머지는 '외 n명'"""
    items = [i for i in (s.strip() for s in items) if i]
    if len(items) <= limit:
        return ", ".join(items)
    return f"{', '.join(items[:limit])} 외 {len(items) - limit}{unit}"


def position(page_kind: str, n: int, total: int, extra: str = "") -> str:
    """위치 표기 (쪽·슬라이드·부·시트 행 범위)"""
    if page_kind == "sheet":
        return extra
    fmt = _POSITION_FORMATS.get(page_kind)
    if not fmt or n <= 0:
        return ""
    return fmt.format(n=n, total=total or "?")


def _join(label: str, fields: list[tuple[str, str]]) -> str:
    """'[레이블] 값 | 키: 값 | …' (빈 값은 생략)"""
    pieces = [f"{key}: {value}" if key else value for key, value in fields if value]
    return f"{label} {' | '.join(pieces)}" if pieces else label


def email_header(meta: dict[str, Any]) -> str:
    """이메일 헤더 (meta: date, sender_name/sender, recipient_names/recipients, cc, subject,
    attachment_filenames)"""
    return _join(
        "[이메일]",
        [
            ("", fmt_datetime(meta.get("date"))),
            ("보낸이", meta.get("sender_name") or meta.get("sender") or ""),
            ("받는이", join_limited(meta.get("recipient_names") or meta.get("recipients") or [])),
            ("참조", join_limited(meta.get("cc_names") or meta.get("cc") or [])),
            ("제목", meta.get("subject") or ""),
            ("첨부", join_limited(meta.get("attachment_filenames") or [], unit="건")),
        ],
    )


def chat_header(meta: dict[str, Any]) -> str:
    """채팅 세션 헤더 (meta: date, date_end, participants, room)"""
    return _join(
        "[채팅]",
        [
            ("", fmt_range(meta.get("date"), meta.get("date_end"))),
            ("참여자", join_limited(meta.get("participants") or [])),
            ("방", meta.get("room") or ""),
        ],
    )


def document_header(meta: dict[str, Any]) -> str:
    """문서 헤더 (meta: filename, page_kind, page, page_count, position_extra, author, last_modified)"""
    return _join(
        "[문서]",
        [
            ("", meta.get("filename") or ""),
            ("", position(meta.get("page_kind", ""), meta.get("page", 0),
                          meta.get("page_count", 0), meta.get("position_extra", ""))),
            ("작성자", meta.get("author") or ""),
            ("수정일", fmt_date(meta.get("last_modified") or meta.get("modified_date") or "")),
        ],
    )


def attachment_header(meta: dict[str, Any]) -> str:
    """첨부 헤더 (meta: filename, 위치, parent_date, parent_sender, parent_recipients, parent_subject)"""
    parent = ""
    if meta.get("parent_unit_id"):
        sender = meta.get("parent_sender") or ""
        recips = meta.get("parent_recipients") or []
        arrow = f"{sender}→{join_limited(recips, limit=1)}" if recips else sender
        subject = meta.get("parent_subject") or ""
        parent = " ".join(
            x for x in (fmt_date(meta.get("parent_date")), arrow, f'"{subject}"' if subject else "")
            if x
        )
    parent_label = "부모 대화" if meta.get("parent_kind") == "chat" else "부모 메일"
    return _join(
        "[첨부]",
        [
            ("", meta.get("filename") or ""),
            ("", position(meta.get("page_kind", ""), meta.get("page", 0),
                          meta.get("page_count", 0), meta.get("position_extra", ""))),
            (parent_label, parent),
        ],
    )


def build_header(unit_type: str, meta: dict[str, Any]) -> str:
    """unit_type 에 맞는 헤더 생성"""
    if unit_type == "email":
        return email_header(meta)
    if unit_type == "chat":
        return chat_header(meta)
    if unit_type == "attachment":
        return attachment_header(meta)
    return document_header(meta)
