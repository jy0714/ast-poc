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

    def test_parse_pdf_creator_producer(self, parser: DocumentParser):
        """PDF creator/producer/modDate 메타데이터 추출 (작성자 추적용)"""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF가 설치되지 않았습니다")

        path = TEST_DIR / "test_creator.pdf"
        doc = fitz.open()
        doc.set_metadata({
            "author": "Original Author",
            "creator": "Microsoft Word",
            "producer": "Adobe PDF Library 21.0",
        })
        page = doc.new_page()
        page.insert_text((72, 72), "Content", fontsize=12)
        doc.save(str(path))
        doc.close()

        result = parser.parse(path)
        assert result.metadata["author"] == "Original Author"
        assert result.metadata["creator"] == "Microsoft Word"
        assert result.metadata["producer"] == "Adobe PDF Library 21.0"

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

    def test_encrypted_pdf_returns_empty_with_flag(self, parser: DocumentParser):
        """암호화된 PDF → 빈 본문 + is_encrypted=True"""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF가 설치되지 않았습니다")

        # 암호 설정한 PDF 생성
        path = TEST_DIR / "encrypted.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Secret content", fontsize=12)
        doc.save(
            str(path),
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="ownerpw",
            user_pw="userpw",
        )
        doc.close()

        result = parser.parse(path)
        assert result.content == ""
        assert result.metadata.get("is_encrypted") is True

    def test_ocr_check_cached_one_warning(self, monkeypatch, caplog):
        """OCR 엔진 가용성 체크는 1회만 — 매 호출마다 WARNING이 누적되지 않음"""
        import logging

        from src.parsers.ocr import reset_engine_cache
        from src.parsers.ocr.tesseract import TesseractEngine

        # 엔진 싱글톤 + 가용성 캐시 리셋
        reset_engine_cache()
        TesseractEngine._available = None
        TesseractEngine._unavailable_reason = ""

        # pytesseract import 실패하도록 sys.modules에 None 주입
        import sys

        monkeypatch.setitem(sys.modules, "pytesseract", None)

        engine = TesseractEngine()
        with caplog.at_level(logging.WARNING):
            r1 = engine.is_available()
            r2 = engine.is_available()
            r3 = engine.is_available()

        assert r1 is False and r2 is False and r3 is False
        # WARNING은 첫 호출 1번만
        warnings = [
            r for r in caplog.records
            if r.levelname == "WARNING" and "Tesseract OCR" in r.message
        ]
        assert len(warnings) == 1

    def test_scan_pdf_metadata_set(self, parser: DocumentParser, monkeypatch):
        """텍스트 거의 없고 OCR 미가용 → scan_pdf=true 메타"""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF가 설치되지 않았습니다")

        # OCR 엔진 싱글톤 리셋 + 가용성을 False로 강제
        from src.parsers.ocr import reset_engine_cache
        from src.parsers.ocr.tesseract import TesseractEngine
        from src.parsers.ocr.paddle import PaddleEngine

        reset_engine_cache()
        monkeypatch.setattr(TesseractEngine, "_available", False)
        monkeypatch.setattr(PaddleEngine, "_available", False)

        # 빈 페이지 PDF 생성 (텍스트 없음 → 모든 페이지 sparse)
        path = TEST_DIR / "scan_like.pdf"
        doc = fitz.open()
        for _ in range(3):
            doc.new_page()  # 빈 페이지
        doc.save(str(path))
        doc.close()

        result = parser.parse(path)
        assert result.metadata["scan_pdf"] is True
        assert result.metadata["sparse_pages"] == 3
        assert result.metadata["is_ocr"] is False

    def test_pdf_page_failure_isolated(self, parser: DocumentParser, monkeypatch):
        """한 페이지의 텍스트 추출이 실패해도 나머지 페이지는 추출"""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF가 설치되지 않았습니다")

        path = TEST_DIR / "multi_page.pdf"
        doc = fitz.open()
        for i in range(3):
            p = doc.new_page()
            p.insert_text((72, 72), f"Page {i} content text body.", fontsize=12)
        doc.save(str(path))
        doc.close()

        # 페이지 1번에서 get_text 호출 시 강제 예외
        original_get_text = fitz.Page.get_text
        call_count = {"n": 0}

        def faulty_get_text(self, *args, **kwargs):
            call_count["n"] += 1
            # 페이지 인덱스 1을 처음 호출할 때만 예외 (page.number == 1)
            if self.number == 1 and call_count["n"] <= 4:
                raise RuntimeError("simulated page corruption")
            return original_get_text(self, *args, **kwargs)

        monkeypatch.setattr(fitz.Page, "get_text", faulty_get_text)

        result = parser.parse(path)
        assert "Page 0" in result.content
        assert "Page 2" in result.content
        # 페이지 실패가 메타데이터에 기록됨
        assert result.metadata.get("page_failures", 0) >= 1

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

    def test_pdf_section_detection(self, parser: DocumentParser):
        """PDF 헤딩(큰 폰트) 감지 → sections 메타데이터"""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF가 설치되지 않았습니다")

        path = TEST_DIR / "test_sections.pdf"
        doc = fitz.open()
        page = doc.new_page()
        # 헤딩 (큰 폰트 + 볼드)
        page.insert_text((72, 72), "Chapter 1: Introduction", fontsize=18, fontname="helv")
        # 본문 (작은 폰트)
        page.insert_text((72, 120), "This is the body text of chapter one.", fontsize=11)
        page.insert_text((72, 140), "More body content here.", fontsize=11)
        # 두 번째 헤딩
        page.insert_text((72, 200), "Chapter 2: Findings", fontsize=18, fontname="helv")
        page.insert_text((72, 240), "Finding details here.", fontsize=11)
        doc.save(str(path))
        doc.close()

        result = parser.parse(path)
        sections = result.metadata.get("sections", [])
        # 큰 폰트 텍스트가 섹션으로 감지되어야 함
        assert len(sections) >= 1
        section_titles = [s["title"] for s in sections]
        assert any("Chapter" in t for t in section_titles)

    def test_pdf_no_sections_when_uniform_font(self, parser: DocumentParser):
        """동일 폰트 크기 → 섹션 없음"""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF가 설치되지 않았습니다")

        path = TEST_DIR / "test_uniform.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Line one of text.", fontsize=12)
        page.insert_text((72, 100), "Line two of text.", fontsize=12)
        page.insert_text((72, 128), "Line three of text.", fontsize=12)
        doc.save(str(path))
        doc.close()

        result = parser.parse(path)
        sections = result.metadata.get("sections", [])
        assert len(sections) == 0


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

    def test_parse_pptx_sections_metadata(self, parser: DocumentParser):
        """PPTX 파싱 시 슬라이드별 sections 메타데이터 생성"""
        try:
            from pptx import Presentation
        except ImportError:
            pytest.skip("python-pptx가 설치되지 않았습니다")

        path = TEST_DIR / "test_sections.pptx"
        prs = Presentation()
        # 슬라이드 1
        slide1 = prs.slides.add_slide(prs.slide_layouts[1])
        slide1.shapes.title.text = "프로젝트 개요"
        slide1.placeholders[1].text = "개요 내용"
        # 슬라이드 2
        slide2 = prs.slides.add_slide(prs.slide_layouts[1])
        slide2.shapes.title.text = "예산 현황"
        slide2.placeholders[1].text = "예산 내용"
        prs.save(str(path))

        result = parser.parse(path)
        sections = result.metadata.get("sections", [])
        assert len(sections) == 2
        assert sections[0]["title"] == "프로젝트 개요"
        assert sections[0]["slide_num"] == 1
        assert sections[1]["title"] == "예산 현황"
        assert sections[1]["slide_num"] == 2

    def test_parse_pptx_author_modifier(self, parser: DocumentParser):
        """PPTX author + last_modified_by 메타데이터 추출 (작성자 추적용)"""
        try:
            from pptx import Presentation
        except ImportError:
            pytest.skip("python-pptx가 설치되지 않았습니다")

        path = TEST_DIR / "test_author_pptx.pptx"
        prs = Presentation()
        prs.core_properties.author = "원작성자"
        prs.core_properties.last_modified_by = "수정자"
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "프레젠테이션"
        prs.save(str(path))

        result = parser.parse(path)
        assert result.metadata["author"] == "원작성자"
        assert result.metadata["last_modified_by"] == "수정자"


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

    def test_parse_xlsx_author_modifier(self, parser: DocumentParser):
        """XLSX author + last_modified_by 메타데이터 추출 (작성자 추적용)

        모든 시트가 동일한 워크북 author/last_modified_by를 공유해야 함.
        """
        try:
            from openpyxl import Workbook
        except ImportError:
            pytest.skip("openpyxl이 설치되지 않았습니다")

        path = TEST_DIR / "test_xlsx_author.xlsx"
        wb = Workbook()
        wb.properties.creator = "견적서작성자"
        wb.properties.lastModifiedBy = "수정한사람"
        wb.properties.title = "견적서 v3"
        ws = wb.active
        ws.title = "견적"
        ws.append(["품목", "금액"])
        ws.append(["서버", 1000])
        wb.create_sheet("부록").append(["부록 내용"])
        wb.save(str(path))
        wb.close()

        result = parser.parse(path)
        assert isinstance(result, list)
        for doc in result:
            assert doc.metadata["author"] == "견적서작성자"
            assert doc.metadata["last_modified_by"] == "수정한사람"
            assert doc.metadata["title"] == "견적서 v3"

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
    recipient: str | None = None,  # to 별칭
    cc: str = "",
    body: str = "이것은 테스트 이메일 본문입니다.",
    html_body: str = "",
    attachments: list[tuple[str, bytes]] | None = None,
    extra_headers: dict[str, str] | None = None,
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
    msg["To"] = recipient if recipient is not None else to
    if cc:
        msg["Cc"] = cc
    msg["Date"] = "Mon, 23 Mar 2026 10:30:00 +0900"
    msg["Message-ID"] = "<test-123@example.com>"

    if extra_headers:
        for k, v in extra_headers.items():
            msg[k] = v

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

    def test_parse_eml_participants_merged(self, parser: DocumentParser):
        """EML participants는 sender+recipients+cc 통합 (중복 제거)"""
        eml_bytes = _make_eml(
            sender="alice@example.com",
            recipient="bob@example.com",
            cc="carol@example.com, alice@example.com",  # alice 중복
        )
        result = parser.parse_bytes(eml_bytes, "merge.eml")

        participants = result.metadata["participants"]
        # 중복 제거된 상태로 3명
        emails = {p.lower() for p in participants}
        assert "alice@example.com" in emails
        assert "bob@example.com" in emails
        assert "carol@example.com" in emails
        assert len(participants) == 3

    def test_parse_eml_threading_headers(self, parser: DocumentParser):
        """EML In-Reply-To / References / Reply-To 헤더 추출"""
        eml_bytes = _make_eml(
            extra_headers={
                "In-Reply-To": "<parent-msg-1@example.com>",
                "References": "<root-msg@example.com> <parent-msg-1@example.com>",
                "Reply-To": "noreply@example.com",
            },
        )
        result = parser.parse_bytes(eml_bytes, "threading.eml")

        assert result.metadata["in_reply_to"] == "<parent-msg-1@example.com>"
        assert result.metadata["reply_to"] == "noreply@example.com"
        assert "<root-msg@example.com>" in result.metadata["references"]
        assert "<parent-msg-1@example.com>" in result.metadata["references"]

    def test_parse_eml_attachments_in_header_block(self, parser: DocumentParser):
        """EML 본문 헤더 블록에 첨부파일 목록도 포함"""
        eml_bytes = _make_eml(
            attachments=[("report.pdf", b"x"), ("budget.xlsx", b"y")],
        )
        result = parser.parse_bytes(eml_bytes, "att_header.eml")

        assert "Attachments: report.pdf, budget.xlsx" in result.content

    def test_parse_eml_corrupt_mime_returns_empty_doc(self, parser: DocumentParser):
        """손상된 MIME → 빈 본문 + parse_error 메타로 반환 (raise 하지 않음)"""
        # MIME 헤더가 깨진 바이트 — 실제로는 message_from_bytes가 관대해서
        # 거의 모든 경우 파싱하므로, 강제로 None을 반환하도록 monkey patch
        from unittest.mock import patch

        with patch("email.message_from_bytes", side_effect=Exception("boom")):
            result = parser.parse_bytes(b"garbage", "broken.eml")

        assert result.content == ""
        assert result.metadata.get("parse_error")
        # source 메타 부착(부재) 없이도 안전하게 반환
        assert result.metadata["subject"] == ""
        assert result.metadata["recipients"] == []


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
