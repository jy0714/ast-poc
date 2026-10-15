"""다국어 이메일 인용(회신 이력) 제거 — AuditRAG v4 참조 구현.

프로젝트에 이미 이메일 본문 정제 함수가 있으면 그 함수에 아래 패턴을 합친다(중복 구현 금지).
규칙: 인용 시작으로 판단되는 첫 위치에서 본문을 자른다. 자른 결과가 비면 원문을 돌려준다.
짧은 회신("OK")도 그대로 하나의 증거 단위가 된다. 인용된 이전 메시지는 각자의 단위로 이미 색인되어 있다.

인용 시작 판단 기준
1) 구분선: -----Original Message----- 및 현지어 변형
2) "On ... wrote:" 류 한 줄 머리말(영·스·체·독·불·중·한)
3) Outlook 헤더 블록: 줄 맨 앞의 헤더 레이블(From:/보낸 사람:/Od:/De:/发件人: …)이
   연속 2줄 이상 나타나는 첫 위치. 'De:'처럼 일반 단어와 겹치는 레이블 때문에 1줄로는 판단하지 않는다.
4) '>' 로 시작하는 인용 줄은 위치와 관계없이 제거
"""

from __future__ import annotations

import re

_SEPARATORS = [
    r"-{2,}\s*Original Message\s*-{2,}",
    r"-{2,}\s*원본 메시지\s*-{2,}",
    r"-{2,}\s*Původní zpráva\s*-{2,}",
    r"-{2,}\s*Mensaje original\s*-{2,}",
    r"-{2,}\s*Ursprüngliche Nachricht\s*-{2,}",
    r"-{2,}\s*Message d'origine\s*-{2,}",
    r"-{2,}\s*原始邮件\s*-{2,}",
    r"-{2,}\s*原始郵件\s*-{2,}",
    r"_{10,}",  # Outlook 웹/모바일의 밑줄 구분선
]

_WROTE_LINES = [
    r"^On .{4,200}wrote:\s*$",
    r"^El .{4,200}escribió:\s*$",
    r"^Dne .{4,200}napsal(?:\(a\)|a)?:\s*$",
    r"^Am .{4,200}schrieb .{0,200}:\s*$",
    r"^Le .{4,200}a écrit\s*:\s*$",
    r"^在.{2,200}写道[:：]\s*$",
    r"^.{2,100}님이 작성:\s*$",
]

# 줄 맨 앞 헤더 레이블(콜론은 전각 포함)
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

_SEP_RE = re.compile("|".join(_SEPARATORS), re.IGNORECASE | re.MULTILINE)
_WROTE_RE = re.compile("|".join(_WROTE_LINES), re.IGNORECASE | re.MULTILINE)
_HEADER_LINE_RE = re.compile(
    r"^\s*\*?(?:" + "|".join(re.escape(x) for x in sorted(_HEADER_LABELS, key=len, reverse=True)) + r")\*?\s*[:：]",
    re.IGNORECASE,
)


def _first_header_block(lines: list[str]) -> int | None:
    """헤더 레이블 줄이 연속 2줄 이상 나오는 첫 줄 번호."""
    for i in range(len(lines) - 1):
        if _HEADER_LINE_RE.match(lines[i]) and _HEADER_LINE_RE.match(lines[i + 1]):
            return i
    return None


def strip_quoted(body: str) -> str:
    """회신 인용 이력을 제거한 본문. 결과가 비면 원문 반환."""
    if not body:
        return body
    text = body.replace("\r\n", "\n")
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

    kept = text[:cut]
    kept = "\n".join(line for line in kept.split("\n") if not line.lstrip().startswith(">"))
    kept = kept.strip()
    return kept if kept else body.strip()
