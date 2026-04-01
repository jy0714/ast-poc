"""스마트 청킹 엔진 유닛 테스트

5종 청커(Document, Chat, Email, Attachment, FixedSize) 로직과 메타데이터 전파를 검증.
"""

from datetime import datetime, timedelta

import pytest

from src.chunkers.chunker import (
    AttachmentChunker,
    ChatChunker,
    Chunk,
    DocumentChunker,
    EmailChunker,
    FixedSizeChunker,
)
from src.parsers.pst_parser import Attachment, ChatMessage, EmailMessage


# === Chunk 데이터 모델 테스트 ===


class TestChunk:
    def test_auto_id_generation(self):
        chunk = Chunk(content="hello", source_type="document")
        assert chunk.chunk_id
        assert len(chunk.chunk_id) == 16

    def test_deterministic_id(self):
        """동일 콘텐츠 + 메타데이터 → 동일 ID"""
        c1 = Chunk(content="hello", source_type="document", metadata={"k": "v"})
        c2 = Chunk(content="hello", source_type="document", metadata={"k": "v"})
        assert c1.chunk_id == c2.chunk_id

    def test_different_content_different_id(self):
        c1 = Chunk(content="hello", source_type="document")
        c2 = Chunk(content="world", source_type="document")
        assert c1.chunk_id != c2.chunk_id

    def test_explicit_id_preserved(self):
        chunk = Chunk(content="hello", chunk_id="custom_id")
        assert chunk.chunk_id == "custom_id"


# === DocumentChunker 테스트 ===


class TestDocumentChunker:
    def test_basic_chunking(self):
        """기본 텍스트 분할"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "가나다라마바사. " * 20  # 약 160자
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        for c in chunks:
            assert c.source_type == "document"
            assert c.content

    def test_short_text_single_chunk(self):
        """짧은 텍스트는 단일 청크"""
        chunker = DocumentChunker(chunk_size=1000)
        chunks = chunker.chunk("짧은 텍스트입니다.")
        assert len(chunks) == 1
        assert chunks[0].content == "짧은 텍스트입니다."

    def test_empty_text_no_chunks(self):
        """빈 텍스트는 빈 리스트"""
        chunker = DocumentChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []
        assert chunker.chunk(None) == []

    def test_metadata_propagation(self):
        """메타데이터가 모든 청크에 전파"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        meta = {"filename": "test.pdf", "author": "홍길동"}
        text = "가나다라마바사아자차. " * 20
        chunks = chunker.chunk(text, metadata=meta)
        for c in chunks:
            assert c.metadata["filename"] == "test.pdf"
            assert c.metadata["author"] == "홍길동"
            assert "chunk_index" in c.metadata
            assert "total_chunks" in c.metadata

    def test_chunk_index_sequential(self):
        """chunk_index가 순차적으로 증가"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "테스트 문장입니다. " * 30
        chunks = chunker.chunk(text)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_parsed_document(self):
        """ParsedDocument 객체 직접 청킹"""
        from src.parsers.document_parser import ParsedDocument

        doc = ParsedDocument(
            filename="test.pdf",
            content="문서 내용입니다. " * 50,
            file_type="pdf",
            metadata={"author": "작성자", "page_count": 5},
        )

        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_parsed_document(doc, extra_metadata={"case_id": "C001"})
        assert len(chunks) > 1
        for c in chunks:
            assert c.metadata["author"] == "작성자"
            assert c.metadata["case_id"] == "C001"


# === DocumentChunker 섹션 기반 청킹 테스트 ===


class TestDocumentChunkerSections:
    def test_section_chunking_splits_by_section(self):
        """sections 메타데이터가 있으면 섹션 단위로 분할"""
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        text = "서론 내용입니다.\n\n제1장 감사 범위\n감사 범위 내용이 여기에 있습니다.\n\n제2장 발견 사항\n발견 사항 내용입니다."
        sections = [
            {"title": "제1장 감사 범위", "offset": 11, "page": 0, "font_size": 16.0},
            {"title": "제2장 발견 사항", "offset": 42, "page": 1, "font_size": 16.0},
        ]
        chunks = chunker.chunk(text, metadata={"filename": "report.pdf", "sections": sections})
        assert len(chunks) >= 2
        # 프리앰블 + 섹션들
        titles = [c.metadata.get("section_title", "") for c in chunks]
        assert "(서문)" in titles
        assert "제1장 감사 범위" in titles or "제2장 발견 사항" in titles

    def test_section_title_in_metadata(self):
        """각 청크에 section_title이 포함"""
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=0)
        text = "Section A 내용입니다.\n\nSection B 내용입니다."
        sections = [
            {"title": "Section A", "offset": 0, "page": 0, "font_size": 14.0},
            {"title": "Section B", "offset": 22, "page": 1, "font_size": 14.0},
        ]
        chunks = chunker.chunk(text, metadata={"sections": sections})
        for c in chunks:
            assert "section_title" in c.metadata

    def test_no_sections_falls_back_to_flat(self):
        """sections가 없으면 기존 방식으로 청킹"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "가나다라마바사. " * 20
        chunks = chunker.chunk(text, metadata={"filename": "doc.pdf"})
        assert len(chunks) > 1
        # section_title이 없어야 함
        assert "section_title" not in chunks[0].metadata

    def test_single_section_falls_back_to_flat(self):
        """섹션이 1개뿐이면 기존 방식 사용"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "내용. " * 30
        sections = [{"title": "유일한 섹션", "offset": 0, "page": 0, "font_size": 14.0}]
        chunks = chunker.chunk(text, metadata={"sections": sections})
        assert len(chunks) > 1
        assert "section_title" not in chunks[0].metadata

    def test_pptx_slide_chunking(self):
        """PPTX 슬라이드 단위 섹션 청킹 — slide_num 메타데이터"""
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=0)
        text = "[슬라이드 1]\n프로젝트 개요 내용\n\n[슬라이드 2]\n일정 계획 내용\n\n[슬라이드 3]\n예산 현황"
        sections = [
            {"title": "프로젝트 개요", "slide_num": 1, "offset": 0},
            {"title": "일정 계획", "slide_num": 2, "offset": 22},
            {"title": "예산 현황", "slide_num": 3, "offset": 40},
        ]
        chunks = chunker.chunk(text, metadata={"filename": "deck.pptx", "sections": sections})
        assert len(chunks) >= 3
        slide_nums = [c.metadata.get("slide_num") for c in chunks if "slide_num" in c.metadata]
        assert 1 in slide_nums
        assert 2 in slide_nums

    def test_sections_not_propagated_to_chunk_meta(self):
        """sections 리스트 자체는 개별 청크 메타데이터에 포함되지 않음"""
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=0)
        text = "A 내용\n\nB 내용"
        sections = [
            {"title": "A", "offset": 0, "page": 0, "font_size": 14.0},
            {"title": "B", "offset": 7, "page": 0, "font_size": 14.0},
        ]
        chunks = chunker.chunk(text, metadata={"filename": "test.pdf", "sections": sections})
        for c in chunks:
            assert "sections" not in c.metadata

    def test_large_section_gets_split(self):
        """chunk_size를 초과하는 섹션은 2차 분할"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        sec1 = "짧은 섹션. "
        sec2 = "긴 섹션 내용입니다. " * 20  # ~200자
        text = sec1 + sec2
        sections = [
            {"title": "짧은 섹션", "offset": 0, "page": 0, "font_size": 14.0},
            {"title": "긴 섹션", "offset": len(sec1), "page": 1, "font_size": 14.0},
        ]
        chunks = chunker.chunk(text, metadata={"sections": sections})
        # 긴 섹션이 여러 청크로 분할되어야 함
        long_chunks = [c for c in chunks if c.metadata.get("section_title") == "긴 섹션"]
        assert len(long_chunks) > 1


# === ChatChunker 테스트 ===


class TestChatChunker:
    def _make_messages(
        self,
        count: int,
        gap_minutes: int = 5,
        start: datetime | None = None,
    ) -> list[dict]:
        """테스트용 채팅 메시지 생성"""
        base = start or datetime(2026, 3, 14, 10, 0)
        messages = []
        for i in range(count):
            messages.append({
                "sender": f"user{i % 2}",
                "body": f"메시지 {i}",
                "timestamp": base + timedelta(minutes=i * gap_minutes),
            })
        return messages

    def test_single_window(self):
        """모든 메시지가 한 윈도우 내"""
        chunker = ChatChunker(window_minutes=30)
        msgs = self._make_messages(5, gap_minutes=5)  # 0~20분, 30분 윈도우 내
        chunks = chunker.chunk(msgs)
        assert len(chunks) == 1
        assert chunks[0].source_type == "teams_chat"
        assert chunks[0].metadata["message_count"] == 5

    def test_window_split(self):
        """시간 간격으로 윈도우 분할"""
        chunker = ChatChunker(window_minutes=10)
        msgs = self._make_messages(6, gap_minutes=5)  # 0, 5, 10, 15, 20, 25분
        # 10분 윈도우: [0,5,10] [15,20,25] — 10분 gap at 10→15
        # Actually: gap between msg[2](10min) and msg[3](15min) = 5min < 10 → same group
        # gap between each = 5min, all < 10min → single group
        # Let's make a clearer test
        msgs_with_gap = [
            {"sender": "A", "body": "안녕", "timestamp": datetime(2026, 3, 14, 10, 0)},
            {"sender": "B", "body": "안녕", "timestamp": datetime(2026, 3, 14, 10, 5)},
            # 30분 gap
            {"sender": "A", "body": "점심?", "timestamp": datetime(2026, 3, 14, 10, 40)},
            {"sender": "B", "body": "좋아", "timestamp": datetime(2026, 3, 14, 10, 42)},
        ]
        chunks = chunker.chunk(msgs_with_gap)
        assert len(chunks) == 2
        assert chunks[0].metadata["message_count"] == 2
        assert chunks[1].metadata["message_count"] == 2

    def test_metadata_date_range(self):
        """date_range_start/end 메타데이터"""
        chunker = ChatChunker(window_minutes=60)
        msgs = self._make_messages(3, gap_minutes=10)
        chunks = chunker.chunk(msgs, metadata={"conversation_id": "conv1"})
        assert len(chunks) == 1
        assert "date_range_start" in chunks[0].metadata
        assert "date_range_end" in chunks[0].metadata
        assert chunks[0].metadata["conversation_id"] == "conv1"

    def test_participants_extracted(self):
        """참여자가 자동 추출"""
        chunker = ChatChunker(window_minutes=60)
        msgs = [
            {"sender": "김감사", "body": "확인", "timestamp": datetime(2026, 3, 14, 10, 0)},
            {"sender": "박대리", "body": "네", "timestamp": datetime(2026, 3, 14, 10, 1)},
            {"sender": "김감사", "body": "감사", "timestamp": datetime(2026, 3, 14, 10, 2)},
        ]
        chunks = chunker.chunk(msgs)
        participants = set(chunks[0].metadata["participants"])
        assert participants == {"김감사", "박대리"}

    def test_empty_messages(self):
        """빈 메시지 리스트"""
        chunker = ChatChunker()
        assert chunker.chunk([]) == []

    def test_format_includes_time(self):
        """포맷에 시간 포함"""
        chunker = ChatChunker(window_minutes=60)
        msgs = [
            {"sender": "A", "body": "안녕하세요", "timestamp": datetime(2026, 3, 14, 10, 30)},
        ]
        chunks = chunker.chunk(msgs)
        assert "[10:30]" in chunks[0].content
        assert "A:" in chunks[0].content

    def test_chunk_chat_messages_groups_by_conversation(self):
        """ChatMessage 객체를 conversation_id별로 그룹핑"""
        chunker = ChatChunker(window_minutes=60)
        chat_msgs = [
            ChatMessage(
                sender="A", body="대화1", timestamp=datetime(2026, 3, 14, 10, 0),
                conversation_id="conv_1", participants=["A", "B"],
            ),
            ChatMessage(
                sender="B", body="대화1 응답", timestamp=datetime(2026, 3, 14, 10, 5),
                conversation_id="conv_1", participants=["A", "B"],
            ),
            ChatMessage(
                sender="C", body="다른대화", timestamp=datetime(2026, 3, 14, 10, 0),
                conversation_id="conv_2", participants=["C", "D"],
            ),
        ]
        chunks = chunker.chunk_chat_messages(chat_msgs)
        assert len(chunks) == 2
        conv_ids = {c.metadata["conversation_id"] for c in chunks}
        assert conv_ids == {"conv_1", "conv_2"}


# === EmailChunker 테스트 ===


class TestEmailChunker:
    def _make_email(
        self,
        subject: str,
        sender: str = "김감사",
        recipients: list[str] | None = None,
        body: str = "이메일 본문",
        date: datetime | None = None,
    ) -> EmailMessage:
        return EmailMessage(
            subject=subject,
            sender=sender,
            recipients=recipients or ["박대리"],
            body=body,
            date=date or datetime(2026, 3, 14, 10, 0),
        )

    def test_thread_grouping(self):
        """Re:/Fwd: 접두사 제거로 스레드 그룹핑"""
        chunker = EmailChunker(max_chars=5000)
        emails = [
            self._make_email("감사 보고서 검토", date=datetime(2026, 3, 14, 9, 0)),
            self._make_email("Re: 감사 보고서 검토", date=datetime(2026, 3, 14, 10, 0)),
            self._make_email("Fwd: Re: 감사 보고서 검토", date=datetime(2026, 3, 14, 11, 0)),
        ]
        chunks = chunker.chunk(emails)
        assert len(chunks) == 1  # 같은 스레드
        assert chunks[0].metadata["email_count"] == 3
        assert chunks[0].source_type == "email"

    def test_separate_threads(self):
        """다른 subject = 다른 스레드"""
        chunker = EmailChunker(max_chars=5000)
        emails = [
            self._make_email("감사 보고서"),
            self._make_email("비용 분석 요청"),
        ]
        chunks = chunker.chunk(emails)
        assert len(chunks) == 2

    def test_long_thread_split(self):
        """긴 스레드는 2차 분할"""
        chunker = EmailChunker(max_chars=200)
        long_body = "매우 긴 이메일 본문입니다. " * 50
        emails = [
            self._make_email("긴 스레드", body=long_body),
            self._make_email("Re: 긴 스레드", body=long_body),
        ]
        chunks = chunker.chunk(emails)
        assert len(chunks) > 1
        for c in chunks:
            assert c.source_type == "email"
            assert c.metadata["thread_subject"] == "긴 스레드"

    def test_thread_metadata(self):
        """스레드 메타데이터 (participants, date_range, has_attachments)"""
        chunker = EmailChunker(max_chars=5000)
        emails = [
            EmailMessage(
                subject="보고서",
                sender="김감사",
                recipients=["박대리", "이과장"],
                body="검토 바랍니다.",
                date=datetime(2026, 3, 14, 9, 0),
                attachments=[Attachment(filename="a.pdf", content=b"data")],
            ),
            EmailMessage(
                subject="Re: 보고서",
                sender="박대리",
                recipients=["김감사"],
                body="확인했습니다.",
                date=datetime(2026, 3, 14, 14, 0),
            ),
        ]
        chunks = chunker.chunk(emails)
        meta = chunks[0].metadata
        assert "김감사" in meta["participants"]
        assert "박대리" in meta["participants"]
        assert meta["has_attachments"] is True
        assert "date_range_start" in meta
        assert "date_range_end" in meta

    def test_empty_subject(self):
        """제목이 없는 이메일"""
        chunker = EmailChunker(max_chars=5000)
        emails = [self._make_email("")]
        chunks = chunker.chunk(emails)
        assert len(chunks) == 1
        assert chunks[0].metadata["thread_subject"] == "(제목 없음)"

    def test_empty_emails(self):
        """빈 이메일 리스트"""
        chunker = EmailChunker()
        assert chunker.chunk([]) == []

    def test_chronological_order_in_thread(self):
        """스레드 내 이메일이 시간순으로 정렬"""
        chunker = EmailChunker(max_chars=5000)
        emails = [
            self._make_email("보고서", body="두번째", date=datetime(2026, 3, 14, 14, 0)),
            self._make_email("Re: 보고서", body="첫번째", date=datetime(2026, 3, 14, 9, 0)),
        ]
        chunks = chunker.chunk(emails)
        content = chunks[0].content
        # "첫번째"가 "두번째"보다 먼저 나와야 함
        assert content.index("첫번째") < content.index("두번째")


# === AttachmentChunker 테스트 ===


class TestAttachmentChunker:
    def test_chunk_txt_attachment(self):
        """TXT 첨부파일 청킹"""
        chunker = AttachmentChunker(chunk_size=50, chunk_overlap=10)
        att = Attachment(
            filename="memo.txt",
            content=("이것은 첨부파일 텍스트입니다. " * 10).encode("utf-8"),
        )
        chunks = chunker.chunk(
            [att],
            source_metadata={"email_subject": "보고서"},
        )
        assert len(chunks) >= 1
        for c in chunks:
            assert c.source_type == "attachment"
            assert c.metadata["attachment_filename"] == "memo.txt"

    def test_unsupported_file_skipped(self):
        """지원하지 않는 파일은 건너뜀"""
        chunker = AttachmentChunker()
        att = Attachment(filename="image.png", content=b"\x89PNG")
        chunks = chunker.chunk([att])
        assert len(chunks) == 0

    def test_empty_attachments(self):
        """빈 첨부파일 리스트"""
        chunker = AttachmentChunker()
        assert chunker.chunk([]) == []


# === FixedSizeChunker ===


class TestFixedSizeChunker:
    def test_basic_chunking(self):
        """기본 고정 크기 분할"""
        chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=2)
        text = " ".join(f"word{i}" for i in range(12))
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.source_type == "document"
            assert c.metadata["chunking_method"] == "fixed"

    def test_overlap(self):
        """오버랩 토큰이 다음 청크 시작에 포함"""
        chunker = FixedSizeChunker(chunk_size=4, chunk_overlap=2)
        text = "a b c d e f g h"
        chunks = chunker.chunk(text)
        # chunk_size=4, step=2: [a,b,c,d], [c,d,e,f], [e,f,g,h], [g,h]
        assert len(chunks) >= 3
        # 두 번째 청크에 첫 번째 청크의 마지막 토큰 포함
        second_tokens = chunks[1].content.split()
        first_tokens = chunks[0].content.split()
        # 오버랩 확인: 두 번째 청크의 처음 토큰이 첫 번째 청크에 존재
        assert second_tokens[0] in first_tokens

    def test_empty_text(self):
        """빈 텍스트"""
        chunker = FixedSizeChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_short_text_single_chunk(self):
        """짧은 텍스트는 단일 청크"""
        chunker = FixedSizeChunker(chunk_size=512, chunk_overlap=128)
        chunks = chunker.chunk("짧은 텍스트")
        assert len(chunks) == 1

    def test_metadata_propagation(self):
        """메타데이터 전파"""
        chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
        text = " ".join(f"w{i}" for i in range(10))
        chunks = chunker.chunk(text, metadata={"filename": "test.pdf"})
        for c in chunks:
            assert c.metadata["filename"] == "test.pdf"
            assert "chunk_index" in c.metadata

    def test_default_source_type_document(self):
        """기본 source_type은 document"""
        chunker = FixedSizeChunker(chunk_size=3, chunk_overlap=0)
        chunks = chunker.chunk("a b c d e f")
        for c in chunks:
            assert c.source_type == "document"

    def test_custom_source_type_email(self):
        """source_type='email' 전달 시 보존"""
        chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
        chunks = chunker.chunk("word " * 10, source_type="email")
        assert len(chunks) >= 1
        for c in chunks:
            assert c.source_type == "email"

    def test_custom_source_type_teams_chat(self):
        """source_type='teams_chat' 전달 시 보존"""
        chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
        chunks = chunker.chunk("msg " * 10, source_type="teams_chat")
        for c in chunks:
            assert c.source_type == "teams_chat"

    def test_custom_source_type_attachment(self):
        """source_type='attachment' 전달 시 보존"""
        chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
        chunks = chunker.chunk("data " * 10, source_type="attachment")
        for c in chunks:
            assert c.source_type == "attachment"
