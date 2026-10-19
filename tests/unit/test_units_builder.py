"""증거 단위 빌더 테스트 (src/units/builder.py)

지시서 §5.10: ID 결정성, 중복 메일 제거, 채팅 30분 세션·방 키, 첨부 부모 연결(PST·EML),
헤더 형식, 파트 분할, XLSX 행 블록(머리글 반복), DOCX 가상 페이지, PDF 빈 페이지 번호 유지
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage as PyEmail
from pathlib import Path

import pytest

from src.parsers.document_parser import DocumentParser
from src.parsers.pst_parser import (
    Attachment,
    ChatMessage,
    EmailMessage,
    PSTParseResult,
    chat_room_key,
)
from src.units.builder import (
    UnitBuilder,
    UnitCollector,
    make_attachment_id,
    make_chat_id,
    make_email_id,
)
from src.units.tokens import TokenCounter

# 문자 1개 = 토큰 1개 근사 카운터 (빠르고 결정적)
COUNTER = TokenCounter(hf_name=None, chars_per_token=1.0)


class _NoOcr:
    name = "none"

    def is_available(self) -> bool:
        return False


@pytest.fixture
def builder(monkeypatch: pytest.MonkeyPatch) -> UnitBuilder:
    parser = DocumentParser(ocr_languages="eng")
    monkeypatch.setattr(parser, "_get_ocr_engine", lambda: _NoOcr())
    return UnitBuilder("case1", counter=COUNTER, max_part_tokens=200, doc_parser=parser,
                       clean_enabled=True, gap_minutes=30)


def _eml(message_id: str = "<A1@x.com>", body: str = "검토 결과 회신드립니다.",
         subject: str = "RE: 3월 정산 건", attachments: list[tuple[str, bytes, str, str]] | None = None,
         to: str = "김철수 <cs@x.com>, 이영희 <yh@x.com>") -> bytes:
    m = PyEmail()
    m["From"] = "홍길동 <gd@x.com>"
    m["To"] = to
    m["Cc"] = "박민수 <ms@x.com>"
    m["Subject"] = subject
    m["Date"] = "Tue, 05 Mar 2024 05:22:00 +0000"
    if message_id:
        m["Message-ID"] = message_id
    m.set_content(body)
    for name, data, maintype, subtype in attachments or []:
        m.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)
    return m.as_bytes()


def _write(tmp_path: Path, rel: str, data: bytes) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


# ─────────────── ID ───────────────


def test_email_id_rules():
    assert make_email_id("<ABC@x.com>") == make_email_id("<abc@x.com>  ")
    assert make_email_id("<abc@x.com>").startswith("em:") and len(make_email_id("x")) == 19
    fb1 = make_email_id("", "a", "2024", "s", "body")
    assert fb1 == make_email_id("", "a", "2024", "s", "body") != make_email_id("", "b", "2024", "s", "body")
    start = datetime(2024, 3, 5, 14, 0, tzinfo=timezone(timedelta(hours=9)))
    assert make_chat_id("room", start).endswith(":20240305T050000")
    assert make_attachment_id("em:0123456789abcdef", 2, ":p3") == "att:0123456789abcdef:2:p3"


def test_ids_deterministic_across_rebuilds(builder: UnitBuilder, tmp_path: Path):
    p = _write(tmp_path, "a/m.eml", _eml(attachments=[("q.txt", b"line1\nline2", "text", "plain")]))
    ids1 = [u.unit_id for u in builder.build_file(p, "a/m.eml")]
    ids2 = [u.unit_id for u in builder.build_file(p, "a/m.eml")]
    assert ids1 == ids2
    assert ids1[0] == make_email_id("<A1@x.com>")
    assert ids1[1] == make_attachment_id(ids1[0], 1, ":b1")


# ─────────────── 이메일 · 헤더 · 정제 ───────────────


def test_email_header_format_and_timezone(builder: UnitBuilder, tmp_path: Path):
    to = ", ".join(f"사람{i} <p{i}@x.com>" for i in range(7))
    p = _write(tmp_path, "m.eml", _eml(to=to, attachments=[("q.txt", b"x" * 30, "text", "plain")]))
    email = builder.build_file(p, "m.eml")[0]
    # 05:22 UTC → 14:22 KST, 받는이 7명 → 앞 5명 + 외 2명
    assert email.header.startswith("[이메일] 2024-03-05 14:22 | 보낸이: 홍길동 | 받는이: 사람0")
    assert "사람4 외 2명" in email.header
    assert "| 참조: 박민수 | 제목: RE: 3월 정산 건 | 첨부: q.txt" in email.header
    assert email.meta["date"] == "2024-03-05T05:22:00+00:00"  # 메타는 원본 그대로
    assert email.meta["case_id"] == "case1" and email.meta["lang"] == "ko"


def test_reply_quote_stripped_but_forward_kept(builder: UnitBuilder, tmp_path: Path):
    quoted = "확인했습니다.\n\nFrom: 김철수\nSent: Monday\nTo: 홍길동\nSubject: 정산\n\n이전 메일 본문"
    reply = builder.build_file(_write(tmp_path, "r.eml", _eml(body=quoted)), "r.eml")[0]
    assert reply.text == "확인했습니다."
    fwd = builder.build_file(
        _write(tmp_path, "f.eml", _eml("<F1@x.com>", body=quoted, subject="FW: 정산")), "f.eml"
    )[0]
    assert "이전 메일 본문" in fwd.text and fwd.meta["forwarded"] is True
    assert builder.stats.counts["forwarded_emails"] == 1


def test_long_email_split_into_parts_with_same_header(builder: UnitBuilder, tmp_path: Path):
    body = "\n".join(f"{i:03d} 결재 절차 확인 요청드립니다 금액 검토 필요" for i in range(40))
    u = builder.build_file(_write(tmp_path, "l.eml", _eml(body=body)), "l.eml")[0]
    assert len(u.parts) > 1
    assert all(p.n_tokens <= 200 for p in u.parts)
    assert {p.header for p in u.parts} == {u.header}
    assert [p.part_id for p in u.parts] == [f"{u.unit_id}#{k}" for k in range(1, len(u.parts) + 1)]
    assert builder.stats.counts["split_units:email"] == 1


# ─────────────── 첨부 부모 연결 ───────────────


def test_eml_attachment_linked_to_parent(builder: UnitBuilder, tmp_path: Path):
    p = _write(tmp_path, "m.eml", _eml(attachments=[
        ("견적서.txt", "단가 1,200,000원".encode(), "text", "plain"),
        ("photo.png", b"\x89PNG....", "image", "png"),
    ]))
    units = builder.build_file(p, "m.eml")
    email, att = units[0], units[1]
    assert len(units) == 2  # 이미지는 단위 없음
    assert att.unit_type == "attachment"
    assert att.meta["parent_unit_id"] == email.unit_id
    assert att.meta["parent_subject"] == "RE: 3월 정산 건"
    assert att.header.startswith("[첨부] 견적서.txt | 1/1부 | 부모 메일: 2024-03-05 홍길동→김철수 외 1명")
    assert builder.stats.excluded_types[".png"] == 1


def test_pst_attachments_linked_and_orphans(builder: UnitBuilder):
    linked = Attachment("a.txt", "첨부 본문 내용입니다".encode())
    orphan = Attachment("o.txt", "부모 없는 첨부 내용".encode())
    em = EmailMessage(subject="견적", sender="홍길동", recipients=["김철수"], body="첨부 확인 바랍니다",
                      date=datetime(2024, 3, 5, 9, 0), attachments=[linked], message_id="<P1@x>",
                      folder_path="Inbox")
    result = PSTParseResult(emails=[em], attachments=[linked, orphan])
    units = builder.build_from_pst(result, "box.pst")
    by_type = {u.unit_type: [] for u in units}
    for u in units:
        by_type[u.unit_type].append(u)
    atts = by_type["attachment"]
    assert len(atts) == 2
    linked_u = next(u for u in atts if u.meta["filename"] == "a.txt")
    orphan_u = next(u for u in atts if u.meta["filename"] == "o.txt")
    assert linked_u.meta["parent_unit_id"] == make_email_id("<P1@x>")
    assert orphan_u.meta["parent_unit_id"] == ""
    assert "부모 메일" not in orphan_u.header
    assert builder.stats.counts["orphan_attachments"] == 1


# ─────────────── 중복 제거 ───────────────


def test_duplicate_email_merged(builder: UnitBuilder, tmp_path: Path):
    data = _eml(attachments=[("q.txt", b"same attachment", "text", "plain")])
    a = builder.build_file(_write(tmp_path, "custA/Inbox/m.eml", data), "custA/Inbox/m.eml")
    b = builder.build_file(_write(tmp_path, "custB/Sent/m.eml", data), "custB/Sent/m.eml")
    col = UnitCollector()
    col.add(a)
    col.add(b)
    assert len(col.units) == 2
    assert col.duplicates == {"email": 1, "attachment": 1}
    email = col.units[make_email_id("<A1@x.com>")]
    assert email.meta["folder_paths"] == ["custA/Inbox", "custB/Sent"]
    assert email.parts[0].meta["folder_paths"] == ["custA/Inbox", "custB/Sent"]


# ─────────────── 채팅 ───────────────


def _chat(sender: str, minute: int, topic: str = "정산팀", folder: str = "TeamsMessagesData",
          participants: list[str] | None = None) -> ChatMessage:
    return ChatMessage(
        sender=sender, body=f"{sender} 메시지 {minute}", timestamp=datetime(2024, 3, 5, 14, 0) +
        timedelta(minutes=minute), conversation_id=chat_room_key(folder, topic),
        participants=participants or [sender], conversation_topic=topic, folder_path=folder,
    )


def test_chat_sessions_gap_and_room_key(builder: UnitBuilder):
    chats = [
        _chat("홍길동", 0), _chat("김철수", 10), _chat("이영희", 25),  # 세션 1 (참여자 바뀌어도 같은 방)
        _chat("홍길동", 70), _chat("김철수", 80),  # 45분 넘게 쉬고 → 세션 2
        _chat("홍길동", 5, topic="다른방"),  # 다른 방
    ]
    units = builder.build_from_pst(PSTParseResult(chats=chats), "box.pst")
    sessions = [u for u in units if u.unit_type == "chat"]
    assert len(sessions) == 3
    main = sorted((u for u in sessions if u.meta["room"] == "정산팀"), key=lambda u: u.meta["date"])
    assert [u.meta["n_messages"] for u in main] == [3, 2]
    assert main[0].meta["chat_type"] == "group" and main[1].meta["chat_type"] == "1:1"
    assert main[0].unit_id == make_chat_id(chat_room_key("TeamsMessagesData", "정산팀"),
                                           datetime(2024, 3, 5, 14, 0))
    assert main[0].header == "[채팅] 2024-03-05 14:00~14:25 | 참여자: 홍길동, 김철수, 이영희 | 방: 정산팀"
    assert main[0].text.split("\n")[0] == "14:00 홍길동: 홍길동 메시지 0"


def test_chat_room_key_without_topic():
    assert chat_room_key("Teams/Chat", "") == "Teams/Chat"
    assert chat_room_key("Teams/Chat", "정산") == "Teams/Chat__정산"


# ─────────────── 문서 형식별 단위 ───────────────


def test_xlsx_row_blocks_repeat_header(builder: UnitBuilder, tmp_path: Path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "견적"
    ws.append(["품목", "수량", "금액"])
    for i in range(60):
        ws.append([f"부품{i:02d}", i, i * 1000])
    buf = io.BytesIO()
    wb.save(buf)
    units = builder.build_file(_write(tmp_path, "docs/q.xlsx", buf.getvalue()), "docs/q.xlsx")
    assert len(units) > 1
    for u in units:
        assert u.text.startswith("품목\t수량\t금액\n")
        assert u.meta["page_kind"] == "sheet" and len(u.parts) == 1
    assert units[0].meta["row_start"] == 2
    assert units[-1].meta["row_end"] == 61
    assert ":s1r2-" in units[0].unit_id
    assert "| 시트 견적 2~" in units[0].header


def test_docx_virtual_pages(builder: UnitBuilder, tmp_path: Path):
    from docx import Document

    d = Document()
    d.add_heading("1. 목적", level=1)
    d.add_paragraph("이 규정은 정산 절차를 정한다.")
    d.add_heading("2. 범위", level=1)
    for i in range(20):
        d.add_paragraph(f"범위 조항 {i:02d}: 모든 법인 카드 사용 건에 적용한다.")
    buf = io.BytesIO()
    d.save(buf)
    units = builder.build_file(_write(tmp_path, "r.docx", buf.getvalue()), "r.docx")
    assert len(units) >= 3  # 1절 1개 + 길이 때문에 나뉜 2절 여러 개
    assert units[0].text.startswith("1. 목적")
    assert all(u.meta["page_kind"] == "block" and len(u.parts) == 1 for u in units)
    n = len(units)
    assert units[0].header.startswith(f"[문서] r.docx | 1/{n}부")
    assert units[1].unit_id.endswith(":b2")


def test_pdf_empty_page_numbering(builder: UnitBuilder, tmp_path: Path):
    import fitz

    doc = fitz.open()
    for text in ["first page approval text " * 3, "", "third page payment text " * 3]:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    units = builder.build_file(_write(tmp_path, "s.pdf", data), "s.pdf")
    assert [u.meta["page"] for u in units] == [1, 3]
    assert units[1].unit_id.endswith(":p3")
    assert "| 3/3쪽" in units[1].header
    assert builder.stats.counts["empty_pages:pdf"] == 1


def test_ad_email_dropped_with_attachments(builder, tmp_path):
    raw = _eml(subject="(광고) 할인 행사", attachments=[("a.txt", "첨부 본문".encode(), "text", "plain")])
    assert builder.build_file(_write(tmp_path, "m/ad.eml", raw), "m/ad.eml") == []
    assert builder.stats.counts["ads_removed"] == 1
    assert builder.stats.counts["ads_removed_attachments"] == 1
    assert builder.stats.counts["attachments_total"] == 0  # 첨부를 파싱하지 않음


def test_ad_filter_can_be_disabled(tmp_path):
    b = UnitBuilder("case1", counter=COUNTER, max_part_tokens=200, drop_ads=False)
    units = b.build_file(_write(tmp_path, "m/ad.eml", _eml(subject="[광고] 소식")), "m/ad.eml")
    assert [u.unit_type for u in units] == ["email"]
