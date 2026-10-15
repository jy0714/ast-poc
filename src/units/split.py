"""긴 단위 → 파트 분할, XLSX 행 블록, DOCX 가상 페이지 (토큰 기준)

- split_text: UNIT_MAX_PART_TOKENS 를 넘는 본문을 줄 경계에서 나눈다. 한 줄이 상한을 넘으면
  토큰 오프셋으로 강제로 자른다. 이메일·긴 PDF 페이지의 파트가 이렇게 생긴다.
- row_blocks: XLSX/XLS 시트를 "머리글 행 + 데이터 행" 블록으로 묶는다. 블록 하나가 단위다
  (시트 하나가 81만 토큰까지 있어서 시트 단위 판정이 불가능 — v4.1 V5).
- docx_blocks: DOCX 제목 구간(파서의 pages)을 상한 이하로 다시 나눈 가상 페이지.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.units.tokens import TokenCounter


@dataclass(frozen=True)
class TextPiece:
    """분할 결과 조각"""

    text: str
    n_tokens: int


@dataclass(frozen=True)
class RowBlock:
    """XLSX 행 블록 (start_row·end_row 는 원본 시트 행 번호, 1부터)"""

    text: str
    n_tokens: int
    start_row: int
    end_row: int


def _hard_split(line: str, max_tokens: int, counter: TokenCounter) -> list[str]:
    """한 줄을 토큰 오프셋 기준 max_tokens 개씩 강제 분할"""
    offsets = counter.offsets(line)
    if not offsets:
        return [line] if line else []
    out: list[str] = []
    for i in range(0, len(offsets), max_tokens):
        chunk = offsets[i : i + max_tokens]
        start = chunk[0][0] if i else 0
        end = offsets[i + max_tokens][0] if i + max_tokens < len(offsets) else len(line)
        out.append(line[start:end])
    return [s for s in out if s.strip()]


def _pack_lines(lines: list[str], max_tokens: int, counter: TokenCounter) -> list[str]:
    """줄들을 상한 이하 묶음으로 탐욕적으로 합침 (줄바꿈 토큰은 줄당 1개로 근사)"""
    counts = counter.count_many(lines)
    groups: list[str] = []
    buf: list[str] = []
    used = 0
    for line, n in zip(lines, counts, strict=True):
        if n > max_tokens:
            if buf:
                groups.append("\n".join(buf))
                buf, used = [], 0
            groups.extend(_hard_split(line, max_tokens, counter))
            continue
        cost = n + (1 if buf else 0)
        if buf and used + cost > max_tokens:
            groups.append("\n".join(buf))
            buf, used = [line], n
        else:
            buf.append(line)
            used += cost
    if buf:
        groups.append("\n".join(buf))
    return [g for g in groups if g.strip()]


def split_text(text: str, max_tokens: int, counter: TokenCounter) -> list[TextPiece]:
    """본문을 max_tokens 이하 조각으로 분할 (넘지 않으면 조각 1개)

    Args:
        text: 정제된 본문 (헤더 제외)
        max_tokens: 조각당 최대 토큰 수
        counter: 토큰 카운터

    Returns:
        TextPiece 리스트 (빈 본문이면 빈 리스트)
    """
    if not text or not text.strip():
        return []
    n = counter.count(text)
    if n <= max_tokens:
        return [TextPiece(text, n)]

    pieces: list[TextPiece] = []
    for group in _pack_lines(text.split("\n"), max_tokens, counter):
        g_n = counter.count(group)
        if g_n <= max_tokens:
            pieces.append(TextPiece(group, g_n))
        else:  # 줄바꿈 토큰 근사 오차로 아주 조금 넘는 경우 — 한 번 더 강제 분할
            for sub in _hard_split(group, max_tokens, counter):
                pieces.append(TextPiece(sub, counter.count(sub)))
    return pieces


def row_blocks(
    rows: list[str], row_numbers: list[int], max_tokens: int, counter: TokenCounter
) -> list[RowBlock]:
    """시트 행들을 '머리글 행 + 데이터 행' 블록으로 묶음

    머리글 행은 시트의 첫 비어 있지 않은 행으로 본다. 머리글이 상한의 절반을 넘으면
    반복하지 않는다(블록마다 머리글만으로 꽉 차는 것을 막기 위해). 한 행이 상한을 넘으면
    그 행만 강제로 잘라 여러 블록으로 만든다.
    """
    pairs = [(r, n) for r, n in zip(rows, row_numbers or range(1, len(rows) + 1), strict=False)
             if r.strip()]
    if not pairs:
        return []
    header_text, header_row = pairs[0]
    header_n = counter.count(header_text)
    repeat_header = header_n <= max_tokens // 2
    data = pairs[1:]
    if not data:
        return [RowBlock(header_text, header_n, header_row, header_row)]

    counts = counter.count_many([r for r, _ in data])
    budget = max_tokens - (header_n + 1 if repeat_header else 0)
    blocks: list[RowBlock] = []
    buf: list[tuple[str, int]] = []
    used = 0

    def flush() -> None:
        nonlocal buf, used
        if not buf:
            return
        body = "\n".join(r for r, _ in buf)
        text = f"{header_text}\n{body}" if repeat_header else body
        start = header_row if not blocks and not repeat_header else buf[0][1]
        blocks.append(RowBlock(text, counter.count(text), start, buf[-1][1]))
        buf, used = [], 0

    for (row, row_no), n in zip(data, counts, strict=True):
        if n > budget:
            flush()
            for sub in _hard_split(row, max(1, budget), counter):
                text = f"{header_text}\n{sub}" if repeat_header else sub
                blocks.append(RowBlock(text, counter.count(text), row_no, row_no))
            continue
        cost = n + (1 if buf else 0)
        if buf and used + cost > budget:
            flush()
            cost = n
        buf.append((row, row_no))
        used += cost
    flush()
    if not repeat_header and blocks:
        # 머리글을 반복하지 않으면 첫 블록에 머리글 행을 붙여 정보 손실을 막는다
        first = blocks[0]
        text = f"{header_text}\n{first.text}"
        blocks[0] = RowBlock(text, counter.count(text), header_row, first.end_row)
    return blocks


def docx_blocks(sections: list[str], max_tokens: int, counter: TokenCounter) -> list[TextPiece]:
    """DOCX 제목 구간 → 가상 페이지 (구간이 상한을 넘으면 문단 경계에서 다시 나눔)"""
    out: list[TextPiece] = []
    for sec in sections:
        out.extend(split_text(sec, max_tokens, counter))
    return out
