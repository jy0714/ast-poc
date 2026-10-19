"""다국어 이메일 본문 정제 — 회신 인용·서명·면책 문구 제거

feat/search-quality 브랜치의 `chunker.clean_email_body`(한·영 인용, 서명, 면책 문구)와
참조 구현 `docs/v4/ref/email_quote.py`(중·체·스·독·불 인용 헤더, "wrote:" 변형,
헤더 레이블 2줄 연속 규칙)를 하나로 합친 구현이다. 두 구현을 따로 두지 않는다.

규칙
1) 회신 인용: 인용 시작으로 판단되는 첫 위치에서 본문을 자른다.
   - 구분선: -----Original Message----- 및 현지어 변형, 긴 밑줄
   - "On ... wrote:" 류 한 줄 머리말 (영·스·체·독·불·중·한)
   - Outlook 헤더 블록: 줄 맨 앞 헤더 레이블(From:/보낸 사람:/Od:/De:/发件人: …)이
     연속 2줄 이상 나오는 첫 위치. 'De:'처럼 일반 단어와 겹치는 레이블 때문에 1줄로는 판단하지 않는다.
2) 전달 메일 예외(v4.1 V7): 제목이 전달 접두어(FW·Fwd·전달·转发·RV·PŘ·WG)로 시작하면
   인용을 자르지 않는다. 전달된 내용은 이 메일함에 다른 사본이 없는 경우가 많기 때문이다.
3) '>' 인용 줄은 연속 2줄 이상인 블록만 위치와 관계없이 제거 (한 줄은 오탐 위험으로 보존)
4) 서명 구분자("-- ", "Sent from my iPhone") 이후, 면책 문구 이후를 제거
5) 정제 결과가 비면 원문을 돌려준다. 짧은 회신("OK")도 그대로 하나의 증거 단위다.
"""

from __future__ import annotations

import re

# ── 1) 인용 구분선 ──
_SEPARATORS = [
    r"-{2,}\s*Original Message\s*-{2,}",
    r"-{2,}\s*원본\s*(?:메시지|메일)\s*-{2,}",
    r"-{2,}\s*Původní zpráva\s*-{2,}",
    r"-{2,}\s*Mensaje original\s*-{2,}",
    r"-{2,}\s*Ursprüngliche Nachricht\s*-{2,}",
    r"-{2,}\s*Message d'origine\s*-{2,}",
    r"-{2,}\s*原始邮件\s*-{2,}",
    r"-{2,}\s*原始郵件\s*-{2,}",
    r"^_{10,}\s*$",  # Outlook 웹/모바일의 밑줄 구분선 (줄 전체)
]

# ── "On ... wrote:" 류 머리말 ──
_WROTE_LINES = [
    r"^On .{4,200}wrote:\s*$",
    r"^El .{4,200}escribió:\s*$",
    r"^Dne .{4,200}napsal(?:\(a\)|a)?:\s*$",
    r"^Am .{4,200}schrieb .{0,200}:\s*$",
    r"^Le .{4,200}a écrit\s*:\s*$",
    r"^在.{2,200}写道[:：]\s*$",
    r"^.{2,100}님이 작성:\s*$",
]

# ── 줄 맨 앞 헤더 레이블 (콜론은 전각 포함) ──
_HEADER_LABELS = [
    # en
    "From", "Sent", "To", "Cc", "Subject", "Date",
    # ko
    "보낸 사람", "받는 사람", "보낸 날짜", "참조", "제목",
    # cs
    "Od", "Odesláno", "Komu", "Kopie", "Předmět",
    # es
    "De", "Enviado", "Enviado el", "Para", "CC", "Asunto",
    # de
    "Von", "Gesendet", "An", "Betreff",
    # fr
    "Envoyé", "À", "Objet",
    # zh-CN / zh-TW
    "发件人", "发送时间", "收件人", "抄送", "主题",
    "寄件者", "寄件日期", "收件者", "副本", "主旨",
]

# ── 4) 서명·면책 문구 (feat/search-quality clean_email_body에서 이관) ──
_SIGNATURE_PATTERNS = [
    re.compile(r"^-- \s*$", re.MULTILINE),  # RFC 3676 서명 구분자
    re.compile(
        r"^Sent from my (?:iPhone|iPad|Galaxy|Android|mobile)\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
]
_DISCLAIMER_PATTERNS = [
    re.compile(
        r"^This (?:email|message|e-mail) (?:and any attachments? )?(?:is|are) "
        r"(?:intended |)?(?:solely )?(?:for |confidential)",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r"^(?:본\s*(?:메일|이메일|메시지)은?\s*.{0,20}기밀)", re.MULTILINE),
    re.compile(r"^CONFIDENTIALITY\s+(?:NOTICE|WARNING)", re.MULTILINE | re.IGNORECASE),
]

# ── 2) 전달 접두어 (대소문자·전각 콜론 무시) ──
_FORWARD_SUBJECT_RE = re.compile(
    r"^\s*(?:fw|fwd|전달|转发|轉寄|rv|př|wg)\s*[:：]", re.IGNORECASE
)

_SEP_RE = re.compile("|".join(_SEPARATORS), re.IGNORECASE | re.MULTILINE)
_WROTE_RE = re.compile("|".join(_WROTE_LINES), re.IGNORECASE | re.MULTILINE)
_HEADER_LINE_RE = re.compile(
    r"^\s*\*?(?:"
    + "|".join(re.escape(x) for x in sorted(_HEADER_LABELS, key=len, reverse=True))
    + r")\*?\s*[:：]",
    re.IGNORECASE,
)


def is_header_line(line: str) -> bool:
    """줄 맨 앞이 인용 헤더 레이블(From:/To:/보낸 사람:/发件人: …)인지"""
    return bool(_HEADER_LINE_RE.match(line))


# 정보통신망법 제50조: 영리 목적 광고 메일은 제목 맨 앞에 "(광고)"를 표시한다. 괄호 모양만 넓게 받는다.
# 회신·전달(RE:, FW: …)이 앞에 붙은 메일은 사내에서 광고를 두고 오간 대화일 수 있어 남긴다.
_AD_SUBJECT_RE = re.compile(r"^\s*[\[(（【〔<［]\s*광\s*고\s*[\])）】〕>］]")


def is_ad_subject(subject: str) -> bool:
    """제목이 광고 표시("(광고)", "[광고]" 등)로 시작하는지"""
    return bool(subject) and bool(_AD_SUBJECT_RE.match(subject))


def is_forwarded(subject: str) -> bool:
    """제목이 전달 접두어로 시작하는지 (RE: FW: 처럼 회신이 먼저면 False)"""
    return bool(subject) and bool(_FORWARD_SUBJECT_RE.match(subject))


def _first_header_block(lines: list[str]) -> int | None:
    """헤더 레이블 줄이 연속 2줄 이상 나오는 첫 줄 번호"""
    for i in range(len(lines) - 1):
        if _HEADER_LINE_RE.match(lines[i]) and _HEADER_LINE_RE.match(lines[i + 1]):
            return i
    return None


def find_quote_start(text: str) -> int | None:
    """회신 인용이 시작되는 문자 위치 (없으면 None). text 는 \\n 줄바꿈 기준"""
    cut = len(text)
    m = _SEP_RE.search(text)
    if m:
        cut = min(cut, m.start())
    m = _WROTE_RE.search(text)
    if m:
        cut = min(cut, m.start())
    lines = text.split("\n")
    idx = _first_header_block(lines)
    if idx is not None and idx > 0:  # 맨 처음 블록은 파서가 붙인 자기 헤더일 수 있음
        cut = min(cut, sum(len(x) + 1 for x in lines[:idx]))
    return cut if cut < len(text) else None


def _drop_quote_line_blocks(text: str) -> str:
    """'>' 인용 줄이 연속 2줄 이상인 블록만 제거 (한 줄짜리는 오탐 위험이 있어 보존)"""
    lines = text.split("\n")
    kept: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith(">"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith(">"):
                j += 1
            if j - i >= 2:
                i = j
                continue
        kept.append(lines[i])
        i += 1
    return "\n".join(kept)


def _cut_at_first(text: str, patterns: list[re.Pattern[str]]) -> str:
    """패턴 중 가장 먼저 나오는 위치에서 자름"""
    positions = [m.start() for p in patterns if (m := p.search(text))]
    return text[: min(positions)] if positions else text


def strip_quoted(body: str, subject: str = "") -> str:
    """회신 인용·서명·면책 문구를 제거한 본문 (결과가 비면 원문)

    Args:
        body: 메일 본문 (파서가 붙인 From/To 헤더 블록은 제거된 상태)
        subject: 메일 제목 — 전달 메일이면 인용을 자르지 않는다

    Returns:
        정제된 본문
    """
    if not body:
        return body
    text = body.replace("\r\n", "\n")

    if not is_forwarded(subject):
        cut = find_quote_start(text)
        if cut is not None:
            text = text[:cut]

    text = _drop_quote_line_blocks(text)
    text = _cut_at_first(text, _SIGNATURE_PATTERNS)
    text = _cut_at_first(text, _DISCLAIMER_PATTERNS)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if text else body.strip()


def clean_email_body(body: str) -> str:
    """하위 호환: feat/search-quality `chunker.clean_email_body`와 같은 이름의 진입점

    v3.x 청커(EmailChunker)가 계속 쓰는 동안 같은 규칙을 공유하기 위해 둔다.
    제목을 모르므로 전달 메일 예외는 적용되지 않는다.
    """
    return strip_quoted(body)
