"""PST 파서 유닛 테스트

pypff/libratom 없이도 Mock 파서로 전체 데이터 모델과 로직을 검증
"""

import pytest
from datetime import datetime

from src.parsers.pst_parser import (
    Attachment,
    ChatMessage,
    EmailMessage,
    MockPSTParser,
    PSTParseResult,
    _safe_str,
    _strip_html,
)


# === 데이터 모델 테스트 ===


class TestAttachment:
    def test_auto_size(self):
        att = Attachment(filename="test.pdf", content=b"hello world")
        assert att.size == 11

    def test_explicit_size(self):
        att = Attachment(filename="test.pdf", content=b"hello", size=999)
        assert att.size == 999

    def test_empty_content(self):
        att = Attachment(filename="empty.txt", content=b"")
        assert att.size == 0


class TestEmailMessage:
    def test_has_attachments(self):
        msg = EmailMessage(
            subject="테스트",
            sender="a",
            recipients=["b"],
            body="본문",
            date=None,
            attachments=[Attachment(filename="f.pdf", content=b"x")],
        )
        assert msg.has_attachments is True

    def test_no_attachments(self):
        msg = EmailMessage(
            subject="테스트",
            sender="a",
            recipients=["b"],
            body="본문",
            date=None,
        )
        assert msg.has_attachments is False


class TestPSTParseResult:
    def test_summary(self):
        result = PSTParseResult(
            emails=[
                EmailMessage(
                    subject="s", sender="a", recipients=["b"], body="x", date=None
                )
            ],
            chats=[],
            attachments=[],
            total_messages=10,
            total_folders=3,
            errors=["err1"],
        )
        summary = result.summary
        assert summary["total_messages"] == 10
        assert summary["emails"] == 1
        assert summary["chats"] == 0
        assert summary["errors"] == 1


# === 유틸리티 함수 테스트 ===


class TestSafeStr:
    def test_none(self):
        assert _safe_str(None) == ""

    def test_bytes(self):
        assert _safe_str(b"hello") == "hello"

    def test_string(self):
        assert _safe_str("hello") == "hello"

    def test_korean_bytes(self):
        assert _safe_str("한글".encode("utf-8")) == "한글"

    def test_integer(self):
        assert _safe_str(42) == "42"


class TestStripHtml:
    def test_basic(self):
        html = "<p>Hello <b>World</b></p>"
        result = _strip_html(html)
        assert "Hello" in result
        assert "World" in result
        assert "<" not in result

    def test_br_to_newline(self):
        html = "Line1<br>Line2<br/>Line3"
        result = _strip_html(html)
        assert "Line1\nLine2\nLine3" == result

    def test_style_removal(self):
        html = "<style>body{color:red}</style><p>Text</p>"
        result = _strip_html(html)
        assert "color" not in result
        assert "Text" in result

    def test_entities(self):
        html = "&amp; &lt; &gt; &nbsp;"
        result = _strip_html(html)
        assert "&" in result
        assert "<" in result
        assert ">" in result


# === Mock 파서 테스트 ===


class TestMockPSTParser:
    def test_parse_returns_result(self):
        parser = MockPSTParser()
        result = parser.parse()
        assert isinstance(result, PSTParseResult)

    def test_has_emails(self):
        result = MockPSTParser().parse()
        assert len(result.emails) >= 2
        assert result.emails[0].subject == "[내부] Q1 감사 보고서 검토 요청"
        assert result.emails[0].sender == "김감사"

    def test_has_chats(self):
        result = MockPSTParser().parse()
        assert len(result.chats) >= 3
        assert result.chats[0].sender == "김감사"
        assert result.chats[0].chat_type == "1:1"

    def test_email_attachments(self):
        result = MockPSTParser().parse()
        email_with_att = result.emails[0]
        assert email_with_att.has_attachments
        assert email_with_att.attachments[0].filename == "Q1_감사보고서_v2.pdf"

    def test_chat_participants(self):
        result = MockPSTParser().parse()
        chat = result.chats[0]
        assert "김감사" in chat.participants
        assert "박대리" in chat.participants

    def test_chat_conversation_id(self):
        result = MockPSTParser().parse()
        conv_ids = set(c.conversation_id for c in result.chats)
        # 같은 대화의 메시지는 같은 conversation_id를 가져야 함
        assert len(conv_ids) == 1

    def test_summary(self):
        result = MockPSTParser().parse()
        summary = result.summary
        assert summary["total_messages"] == 5
        assert summary["emails"] == 2
        assert summary["chats"] == 3
        assert summary["errors"] == 0

    def test_dates_are_datetime(self):
        result = MockPSTParser().parse()
        for email_msg in result.emails:
            assert isinstance(email_msg.date, datetime)
        for chat in result.chats:
            assert isinstance(chat.timestamp, datetime)
