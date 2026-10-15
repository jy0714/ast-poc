"""이메일 정제 테스트 (src/units/email_clean.py)

- docs/v4/ref/test_ref.py 의 이메일 테스트 이식
- 중국어 인용 헤더 가명 고정 자료 (tests/fixtures/email_zh_quotes.json)
- 전달 메일 예외 (v4.1 V7)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.units.email_clean import find_quote_start, is_forwarded, strip_quoted

FIXTURES = Path(__file__).parent.parent / "fixtures" / "email_zh_quotes.json"


# ---------- ref 이식 ----------


def test_outlook_english_block():
    body = "Please see below.\n\nFrom: Kim <kim@a.com>\nSent: Monday\nTo: Lee\nSubject: RE: test\n\nold text"
    assert strip_quoted(body) == "Please see below."


def test_czech_block_and_separator():
    body = (
        "Souhlasím s návrhem, pošlete fakturu.\n\nOd: Novák Jan\nOdesláno: pondělí\n"
        "Komu: Kim\nPředmět: faktura\n\nstarý text"
    )
    assert strip_quoted(body) == "Souhlasím s návrhem, pošlete fakturu."
    body2 = "Prosím o schválení do pátku.\n-----Původní zpráva-----\nstarý text"
    assert strip_quoted(body2) == "Prosím o schválení do pátku."


def test_spanish_de_single_line_not_cut():
    body = "De: acuerdo con lo hablado, enviamos la factura hoy.\nSaludos"
    assert strip_quoted(body) == body.strip()


def test_spanish_and_chinese_blocks():
    es = (
        "Adjunto la factura corregida para su aprobación.\n\nDe: Ana\nEnviado el: lunes\n"
        "Para: Kim\nAsunto: factura\n\nviejo"
    )
    assert strip_quoted(es) == "Adjunto la factura corregida para su aprobación."
    zh = (
        "请在周五前审批这张发票，谢谢配合。\n\n发件人: 王\n发送时间: 2024年3月5日\n"
        "收件人: Kim\n主题: 发票\n\n旧内容"
    )
    assert strip_quoted(zh) == "请在周五前审批这张发票，谢谢配合。"


def test_wrote_line_and_quote_marks():
    body = (
        "Agreed, go ahead with the payment.\n\n"
        "On Mon, Mar 4, 2024 at 9:00 AM Kim <k@a.com> wrote:\n> old"
    )
    assert strip_quoted(body) == "Agreed, go ahead with the payment."


def test_short_reply_kept_and_empty_returns_original():
    body = "OK\n\nFrom: Kim\nSent: Monday\nTo: Lee\n\nlong old text here"
    assert strip_quoted(body) == "OK"
    only_quote = "> old line 1\n> old line 2"
    assert strip_quoted(only_quote) == only_quote


# ---------- 한국어 Outlook ----------


def test_korean_outlook_block():
    body = (
        "검토 결과 회신드립니다.\n\n보낸 사람: 홍길동\n보낸 날짜: 2024년 3월 5일 화요일 오후 2:22\n"
        "받는 사람: 김철수\n제목: 3월 정산 건\n\n이전 내용"
    )
    assert strip_quoted(body) == "검토 결과 회신드립니다."


def test_signature_and_disclaimer_removed():
    body = "확인했습니다.\n-- \n홍길동 | 재무팀\n"
    assert strip_quoted(body) == "확인했습니다."
    body2 = "Approved.\n\nCONFIDENTIALITY NOTICE: This message is confidential."
    assert strip_quoted(body2) == "Approved."


# ---------- 중국어 가명 고정 자료 ----------


def _zh_cases() -> list[dict]:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _zh_cases(), ids=lambda c: c["id"])
def test_chinese_quote_fixtures(case: dict):
    assert strip_quoted(case["body"], case["subject"]) == case["expected"]


# ---------- 전달 메일 예외 ----------


@pytest.mark.parametrize(
    "subject",
    ["FW: 견적", "Fwd: invoice", "전달: 정산", "转发: 发票", "RV: factura", "PŘ: faktura",
     "WG: Rechnung", "fw：전각 콜론", "  FWD: 앞 공백"],
)
def test_forward_prefixes(subject: str):
    assert is_forwarded(subject)


@pytest.mark.parametrize("subject", ["RE: FW: 견적", "回复: 출장", "Forward planning", ""])
def test_not_forwarded(subject: str):
    assert not is_forwarded(subject)


def test_forwarded_keeps_quote_but_drops_signature_and_quote_lines():
    body = (
        "참고하세요.\n\nFrom: Kim\nSent: Monday\nTo: Lee\nSubject: 견적\n\n원문 견적 내용\n"
        "> 인용 1\n> 인용 2\n-- \n서명"
    )
    out = strip_quoted(body, "FW: 견적")
    assert "From: Kim" in out and "원문 견적 내용" in out
    assert "> 인용 1" not in out and "서명" not in out


def test_find_quote_start_ignores_leading_own_header_block():
    # 첫 줄부터 헤더 블록이면(파서가 붙인 자기 헤더일 수 있음) 인용으로 보지 않는다
    text = "From: Kim\nTo: Lee\n\n본문"
    assert find_quote_start(text) is None
