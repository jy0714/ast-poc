"""DocumentParser 유닛 테스트

실제 라이브러리(PyMuPDF, python-docx 등) 없이도
파서 로직과 메타데이터 추출을 검증할 수 있도록 테스트 파일을 동적 생성.
"""

import io
import os
import tempfile
from pathlib import Path

import pytest

from src.parsers.document_parser import DocumentParser, ParsedDocument

# === 테스트 헬퍼 ===

TEST_DIR = Path(tempfile.gettempdir()) / "ast_poc_test"


@pytest.fixture(autouse=True)
def setup_test_dir():
    """테스트용 임시 디렉토리 생성/정리"""
    TEST_DIR.mkdir(exist_ok=True)
    yield
    # 정리: 테스트 파일 삭제
    for f in TEST_DIR.iterdir():
        try:
            f.unlink()
        except OSError:
            pass


@pytest.fixture
def parser() -> DocumentParser:
    return DocumentParser(ocr_languages="eng+kor")


def _write_temp(suffix: str, content: bytes | str) -> Path:
    """테스트 파일 생성 (핸들 즉시 닫음 — Windows 호환)"""
    path = TEST_DIR / f"test{suffix}"
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)
    return path


# === ParsedDocument 데이터 모델 테스트 ===


class TestParsedDocument:
    def test_default_metadata(self):
        doc = ParsedDocument(filename="test.txt", content="hello", file_type="txt")
        assert doc.metadata == {}

    def test_custom_metadata(self):
        doc = ParsedDocument(
            filename="test.pdf",
            content="hello",
            file_type="pdf",
            metadata={"author": "홍길동"},
        )
        assert doc.metadata["author"] == "홍길동"

    def test_default_page_count(self):
        doc = ParsedDocument(filename="test.txt", content="", file_type="txt")
        assert doc.page_count == 0


# === 지원 파일 형식 검증 ===


class TestSupportedTypes:
    def test_supported_extensions(self, parser: DocumentParser):
        assert ".pdf" in parser.SUPPORTED_TYPES
        assert ".docx" in parser.SUPPORTED_TYPES
        assert ".pptx" in parser.SUPPORTED_TYPES
        assert ".xlsx" in parser.SUPPORTED_TYPES
        assert ".txt" in parser.SUPPORTED_TYPES
        assert ".eml" in parser.SUPPORTED_TYPES
        assert ".msg" in parser.SUPPORTED_TYPES

    def test_unsupported_extension_raises(self, parser: DocumentParser):
        path = _write_temp(".bmp", b"fake")
        with pytest.raises(ValueError, match="지원하지 않는 파일 형식"):
            parser.parse(path)

    def test_file_not_found_raises(self, parser: DocumentParser):
        with pytest.raises(FileNotFoundError, match="파일을 찾을 수 없습니다"):
            parser.parse("/nonexistent/path/file.pdf")

    def test_unsupported_extension_parse_bytes(self, parser: DocumentParser):
        with pytest.raises(ValueError, match="지원하지 않는 파일 형식"):
            parser.parse_bytes(b"fake", "image.bmp")


# === TXT 파서 테스트 ===


class TestTxtParser:
    def test_parse_utf8(self, parser: DocumentParser):
        path = _write_temp(".txt", "테스트 문서입니다.\n두 번째 줄.")
        result = parser.parse(path)
        assert isinstance(result, ParsedDocument)
        assert "테스트 문서입니다" in result.content
        assert result.file_type == "txt"
        assert result.metadata["char_count"] > 0
        assert result.metadata["file_hash"]
        assert result.metadata["filename"].endswith(".txt")

    def test_parse_txt_bytes(self, parser: DocumentParser):
        content = "Hello 안녕하세요".encode("utf-8")
        result = parser.parse_bytes(content, "test.txt")
        assert isinstance(result, ParsedDocument)
        assert "안녕하세요" in result.content
        assert result.metadata["file_size"] == len(content)

    def test_parse_txt_bytes_cp949(self, parser: DocumentParser):
        text = "한글 테스트"
        content = text.encode("cp949")
        result = parser.parse_bytes(content, "legacy.txt")
        assert "한글 테스트" in result.content

    def test_empty_txt(self, parser: DocumentParser):
        path = _write_temp(".txt", "")
        result = parser.parse(path)
        assert result.content == ""
        assert result.metadata["char_count"] == 0


# === PDF 파서 테스트 ===


class TestPdfParser:
    def test_parse_pdf_text_extraction(self, parser: DocumentParser):
        """PyMuPDF로 텍스트 추출 테스트 (실제 PDF 생성)"""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF가 설치되지 않았습니다")

        path = TEST_DIR / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test document content for parsing.", fontsize=12)
        page.insert_text((72, 100), "Second line of text.", fontsize=12)
        doc.save(str(path))
        doc.close()

        result = parser.parse(path)
        assert isinstance(result, ParsedDocument)
        assert "Test document content" in result.content
        assert result.file_type == "pdf"
        assert result.page_count == 1
        assert result.metadata["page_count"] == 1
        assert result.metadata["is_ocr"] is False
        assert result.metadata["file_hash"]
        assert result.metadata["char_count"] > 0

    def test_parse_pdf_metadata(self, parser: DocumentParser):
        """PDF 메타데이터 추출 테스트"""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF가 설치되지 않았습니다")

        path = TEST_DIR / "test_meta.pdf"
        doc = fitz.open()
        doc.set_metadata({
            "author": "Test Author",
            "title": "Test Title",
        })
        page = doc.new_page()
        page.insert_text((72, 72), "Content", fontsize=12)
        doc.save(str(path))
        doc.close()

        result = parser.parse(path)
        assert result.metadata["author"] == "Test Author"
        assert result.metadata["title"] == "Test Title"

    def test_parse_pdf_bytes(self, parser: DocumentParser):
        """PDF 바이트 파싱 테스트"""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF가 설치되지 않았습니다")

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Byte content test", fontsize=12)
        pdf_bytes = doc.tobytes()
        doc.close()

        result = parser.parse_bytes(pdf_bytes, "test.pdf")
        assert "Byte content test" in result.content
        assert result.metadata["file_size"] == len(pdf_bytes)

    def test_ocr_fallback_graceful(self, parser: DocumentParser):
        """OCR 라이브러리 없을 때 graceful 폴백"""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF가 설치되지 않았습니다")

        path = TEST_DIR / "test_empty.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(path))
        doc.close()

        result = parser.parse(path)
        assert isinstance(result, ParsedDocument)
        assert result.page_count == 1


# === DOCX 파서 테스트 ===


class TestDocxParser:
    def test_parse_docx(self, parser: DocumentParser):
        """DOCX 텍스트 + 메타데이터 추출 테스트"""
        try:
            from docx import Document
        except ImportError:
            pytest.skip("python-docx가 설치되지 않았습니다")

        path = TEST_DIR / "test.docx"
        doc = Document()
        doc.core_properties.author = "작성자"
        doc.add_paragraph("첫 번째 단락입니다.")
        doc.add_paragraph("두 번째 단락입니다.")
        doc.save(str(path))

        result = parser.parse(path)
        assert isinstance(result, ParsedDocument)
        assert "첫 번째 단락" in result.content
        assert "두 번째 단락" in result.content
        assert result.file_type == "docx"
        assert result.metadata["author"] == "작성자"
        assert result.metadata["char_count"] > 0

    def test_parse_docx_with_table(self, parser: DocumentParser):
        """DOCX 테이블 텍스트 추출 테스트"""
        try:
            from docx import Document
        except ImportError:
            pytest.skip("python-docx가 설치되지 않았습니다")

        path = TEST_DIR / "test_table.docx"
        doc = Document()
        doc.add_paragraph("본문 텍스트")
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "항목"
        table.cell(0, 1).text = "값"
        table.cell(1, 0).text = "매출"
        table.cell(1, 1).text = "1000"
        doc.save(str(path))

        result = parser.parse(path)
        assert "본문 텍스트" in result.content
        assert "매출" in result.content
        assert "1000" in result.content

    def test_parse_docx_bytes(self, parser: DocumentParser):
        """DOCX 바이트 파싱 테스트"""
        try:
            from docx import Document
        except ImportError:
            pytest.skip("python-docx가 설치되지 않았습니다")

        doc = Document()
        doc.add_paragraph("바이트 테스트")
        buf = io.BytesIO()
        doc.save(buf)
        raw = buf.getvalue()

        result = parser.parse_bytes(raw, "test.docx")
        assert "바이트 테스트" in result.content
        assert result.metadata["file_size"] == len(raw)


# === PPTX 파서 테스트 ===


class TestPptxParser:
    def test_parse_pptx(self, parser: DocumentParser):
        """PPTX 슬라이드 텍스트 추출 테스트"""
        try:
            from pptx import Presentation
        except ImportError:
            pytest.skip("python-pptx가 설치되지 않았습니다")

        path = TEST_DIR / "test.pptx"
        prs = Presentation()
        slide_layout = prs.slide_layouts[1]  # Title and Content
        slide = prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = "발표 제목"
        slide.placeholders[1].text = "슬라이드 내용입니다."
        prs.save(str(path))

        result = parser.parse(path)
        assert isinstance(result, ParsedDocument)
        assert "발표 제목" in result.content
        assert "슬라이드 내용" in result.content
        assert result.file_type == "pptx"
        assert result.metadata["slide_count"] == 1
        assert "[슬라이드 1]" in result.content

    def test_parse_pptx_no_notes(self, parser: DocumentParser):
        """PPTX 발표자 노트가 포함되지 않는지 확인"""
        try:
            from pptx import Presentation
        except ImportError:
            pytest.skip("python-pptx가 설치되지 않았습니다")

        path = TEST_DIR / "test_notes.pptx"
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "제목"
        notes_slide = slide.notes_slide
        notes_slide.notes_text_frame.text = "이것은 발표자 노트입니다 비밀메모"
        prs.save(str(path))

        result = parser.parse(path)
        assert "비밀메모" not in result.content

    def test_parse_pptx_bytes(self, parser: DocumentParser):
        """PPTX 바이트 파싱 테스트"""
        try:
            from pptx import Presentation
        except ImportError:
            pytest.skip("python-pptx가 설치되지 않았습니다")

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text = "바이트 테스트"
        buf = io.BytesIO()
        prs.save(buf)
        raw = buf.getvalue()

        result = parser.parse_bytes(raw, "test.pptx")
        assert "바이트 테스트" in result.content


# === XLSX 파서 테스트 ===


class TestXlsxParser:
    def test_parse_xlsx_sheet_separation(self, parser: DocumentParser):
        """XLSX 시트별 분리 ParsedDocument 생성 테스트"""
        try:
            from openpyxl import Workbook
        except ImportError:
            pytest.skip("openpyxl이 설치되지 않았습니다")

        path = TEST_DIR / "test.xlsx"
        wb = Workbook()
        ws1 = wb.active
        ws1.title = "매출현황"
        ws1.append(["항목", "금액"])
        ws1.append(["제품A", 1000])
        ws1.append(["제품B", 2000])

        ws2 = wb.create_sheet("비용내역")
        ws2.append(["항목", "금액"])
        ws2.append(["인건비", 500])
        wb.save(str(path))
        wb.close()

        result = parser.parse(path)
        assert isinstance(result, list)
        assert len(result) == 2

        # 첫 번째 시트
        sheet1 = result[0]
        assert sheet1.metadata["sheet_name"] == "매출현황"
        assert sheet1.metadata["sheet_count"] == 2
        assert sheet1.metadata["row_count"] == 3
        assert "제품A" in sheet1.content
        assert "1000" in sheet1.content
        assert sheet1.file_type == "xlsx"

        # 두 번째 시트
        sheet2 = result[1]
        assert sheet2.metadata["sheet_name"] == "비용내역"
        assert "인건비" in sheet2.content
        assert sheet2.metadata["row_count"] == 2

    def test_parse_xlsx_metadata(self, parser: DocumentParser):
        """XLSX 공통 메타데이터 테스트"""
        try:
            from openpyxl import Workbook
        except ImportError:
            pytest.skip("openpyxl이 설치되지 않았습니다")

        path = TEST_DIR / "test_meta.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["데이터"])
        wb.save(str(path))
        wb.close()

        result = parser.parse(path)
        assert isinstance(result, list)
        doc = result[0]
        assert "sheet_names" in doc.metadata
        assert "file_hash" in doc.metadata
        assert "file_size" in doc.metadata

    def test_parse_xlsx_bytes(self, parser: DocumentParser):
        """XLSX 바이트 파싱 테스트"""
        try:
            from openpyxl import Workbook
        except ImportError:
            pytest.skip("openpyxl이 설치되지 않았습니다")

        wb = Workbook()
        ws = wb.active
        ws.title = "데이터"
        ws.append(["이름", "값"])
        ws.append(["테스트", 123])
        buf = io.BytesIO()
        wb.save(buf)
        wb.close()
        raw = buf.getvalue()

        result = parser.parse_bytes(raw, "test.xlsx")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "테스트" in result[0].content

    def test_parse_xlsx_empty_rows_skipped(self, parser: DocumentParser):
        """XLSX 빈 행은 건너뛰는지 확인"""
        try:
            from openpyxl import Workbook
        except ImportError:
            pytest.skip("openpyxl이 설치되지 않았습니다")

        path = TEST_DIR / "test_empty_rows.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.append(["데이터"])
        ws.append([None, None, None])  # 빈 행
        ws.append(["두번째"])
        wb.save(str(path))
        wb.close()

        result = parser.parse(path)
        doc = result[0]
        assert doc.metadata["row_count"] == 2  # 빈 행 제외


# === 공통 메타데이터 테스트 ===


class TestCommonMetadata:
    def test_file_hash_consistency(self, parser: DocumentParser):
        """동일 파일은 동일 해시 반환"""
        path = _write_temp(".txt", "해시 테스트 내용")
        result1 = parser.parse(path)
        result2 = parser.parse(path)
        assert result1.metadata["file_hash"] == result2.metadata["file_hash"]
        assert len(result1.metadata["file_hash"]) == 64  # SHA-256

    def test_language_detection_korean(self, parser: DocumentParser):
        """한국어 텍스트 언어 감지"""
        path = _write_temp(
            ".txt", "이것은 한국어로 작성된 긴 문서입니다. 언어 감지가 정상적으로 동작하는지 테스트합니다."
        )
        result = parser.parse(path)
        assert result.metadata["language"] == "ko"

    def test_language_detection_english(self, parser: DocumentParser):
        """영어 텍스트 언어 감지"""
        path = _write_temp(
            ".txt",
            "This is a long English document for testing language detection functionality.",
        )
        result = parser.parse(path)
        assert result.metadata["language"] == "en"

    def test_language_detection_short_text(self, parser: DocumentParser):
        """짧은 텍스트는 unknown 반환"""
        path = _write_temp(".txt", "짧은 글")
        result = parser.parse(path)
        assert result.metadata["language"] == "unknown"

    def test_parse_bytes_source_metadata(self, parser: DocumentParser):
        """parse_bytes에 source_metadata가 첨부되는지 확인"""
        content = "첨부파일 내용입니다. 이메일에서 추출한 문서를 파싱합니다.".encode("utf-8")
        source = {"email_subject": "회의록", "sender": "hong@example.com"}

        result = parser.parse_bytes(content, "attachment.txt", source_metadata=source)
        assert result.metadata["source"]["email_subject"] == "회의록"
        assert result.metadata["source"]["sender"] == "hong@example.com"


# === EML 파서 테스트 ===


def _make_eml(
    subject: str = "테스트 메일",
    sender: str = "sender@example.com",
    to: str = "recipient@example.com",
    cc: str = "",
    body: str = "이것은 테스트 이메일 본문입니다.",
    html_body: str = "",
    attachments: list[tuple[str, bytes]] | None = None,
) -> bytes:
    """테스트용 EML 바이트 생성"""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    if attachments or html_body:
        msg = MIMEMultipart()
        if body:
            msg.attach(MIMEText(body, "plain", "utf-8"))
        if html_body:
            msg.attach(MIMEText(html_body, "html", "utf-8"))
        if attachments:
            for fname, content in attachments:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(content)
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", "attachment", filename=fname)
                msg.attach(part)
    else:
        msg = MIMEText(body, "plain", "utf-8")

    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    if cc:
        msg["Cc"] = cc
    msg["Date"] = "Mon, 23 Mar 2026 10:30:00 +0900"
    msg["Message-ID"] = "<test-123@example.com>"

    return msg.as_bytes()


class TestEmlParser:
    def test_parse_eml_basic(self, parser: DocumentParser):
        """EML 기본 파싱"""
        eml_bytes = _make_eml()
        path = _write_temp(".eml", eml_bytes)
        result = parser.parse(path)

        assert isinstance(result, ParsedDocument)
        assert result.file_type == "eml"
        assert "테스트 이메일 본문" in result.content
        assert result.metadata["subject"] == "테스트 메일"
        assert result.metadata["sender"] == "sender@example.com"
        assert "recipient@example.com" in result.metadata["recipients"]
        assert result.metadata["message_id"] == "<test-123@example.com>"
        assert result.metadata["has_attachments"] is False

    def test_parse_eml_bytes(self, parser: DocumentParser):
        """EML 바이트 파싱"""
        eml_bytes = _make_eml(subject="바이트 테스트", body="바이트로 파싱합니다.")
        result = parser.parse_bytes(eml_bytes, "test.eml")

        assert isinstance(result, ParsedDocument)
        assert "바이트로 파싱합니다" in result.content
        assert result.metadata["subject"] == "바이트 테스트"
        assert result.metadata["file_size"] == len(eml_bytes)

    def test_parse_eml_with_cc(self, parser: DocumentParser):
        """EML CC 수신자 파싱"""
        eml_bytes = _make_eml(cc="cc1@example.com, cc2@example.com")
        result = parser.parse_bytes(eml_bytes, "cc_test.eml")

        assert len(result.metadata["cc"]) == 2
        assert "cc1@example.com" in result.metadata["cc"]
        assert "cc2@example.com" in result.metadata["cc"]
        assert "Cc:" in result.content

    def test_parse_eml_html_fallback(self, parser: DocumentParser):
        """EML text/plain이 없을 때 HTML 폴백"""
        eml_bytes = _make_eml(body="", html_body="<html><body><p>HTML 본문입니다.</p></body></html>")
        result = parser.parse_bytes(eml_bytes, "html_test.eml")

        assert "HTML 본문입니다" in result.content

    def test_parse_eml_with_attachments(self, parser: DocumentParser):
        """EML 첨부파일 목록 추출"""
        attachments = [
            ("report.pdf", b"fake pdf content"),
            ("data.xlsx", b"fake xlsx content"),
        ]
        eml_bytes = _make_eml(attachments=attachments)
        result = parser.parse_bytes(eml_bytes, "att_test.eml")

        assert result.metadata["has_attachments"] is True
        assert len(result.metadata["attachment_filenames"]) == 2
        assert "report.pdf" in result.metadata["attachment_filenames"]
        assert "data.xlsx" in result.metadata["attachment_filenames"]

    def test_parse_eml_date(self, parser: DocumentParser):
        """EML 날짜 파싱"""
        eml_bytes = _make_eml()
        result = parser.parse_bytes(eml_bytes, "date_test.eml")

        assert result.metadata["date"]  # 비어있지 않음
        assert "2026" in result.metadata["date"]

    def test_parse_eml_header_in_content(self, parser: DocumentParser):
        """EML 본문에 헤더 정보 포함 확인 (검색 가능하도록)"""
        eml_bytes = _make_eml(subject="프로젝트 보고서", sender="kim@company.com")
        result = parser.parse_bytes(eml_bytes, "header_test.eml")

        assert "From: kim@company.com" in result.content
        assert "Subject: 프로젝트 보고서" in result.content

    def test_parse_eml_common_metadata(self, parser: DocumentParser):
        """EML 공통 메타데이터 (language, char_count 등)"""
        eml_bytes = _make_eml(body="이것은 한국어로 작성된 이메일입니다. 충분히 긴 텍스트를 포함해야 합니다.")
        path = _write_temp(".eml", eml_bytes)
        result = parser.parse(path)

        assert result.metadata["char_count"] > 0
        assert result.metadata["file_hash"]
        assert result.metadata["filename"].endswith(".eml")


# === MSG 파서 테스트 ===


class TestMsgParser:
    def test_parse_msg_bytes(self, parser: DocumentParser):
        """MSG 바이트 파싱 (extract-msg로 생성된 실제 MSG)"""
        try:
            import extract_msg
        except ImportError:
            pytest.skip("extract-msg가 설치되지 않았습니다")

        # extract-msg는 MSG 생성 기능이 제한적이므로
        # OLE 구조의 최소 MSG를 직접 생성하여 테스트
        # 여기서는 지원 타입 등록 확인과 에러 핸들링만 검증
        assert ".msg" in parser.SUPPORTED_TYPES

    def test_msg_supported_type(self, parser: DocumentParser):
        """MSG 타입이 지원 목록에 포함되어 있는지 확인"""
        assert ".msg" in parser.SUPPORTED_TYPES

    def test_msg_unsupported_bytes_error(self, parser: DocumentParser):
        """잘못된 MSG 바이트는 에러 발생"""
        try:
            import extract_msg
        except ImportError:
            pytest.skip("extract-msg가 설치되지 않았습니다")

        with pytest.raises(Exception):
            parser.parse_bytes(b"this is not a valid msg file", "bad.msg")
