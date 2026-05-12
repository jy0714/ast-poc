"""문서 파서 — PDF, DOCX, PPTX, XLSX, TXT, EML 텍스트 추출

지원 파일 형식:
- PDF: PyMuPDF 텍스트 추출 + Tesseract OCR 폴백 (스캔 문서)
- DOCX: python-docx
- PPTX: python-pptx (발표자 노트 제외)
- XLSX: openpyxl (시트별 구분)
- TXT: 직접 읽기
- EML: Python email 모듈 (RFC 5322)

메타데이터:
- 공통: filename, file_path, file_size, file_type, file_hash, language, char_count
- PDF: author, title, page_count, is_ocr, is_encrypted, created_date, modified_date,
       creator, producer
- DOCX: author, last_modified_by, created_date, last_modified
- PPTX: author, title, slide_count, last_modified_by, created_date, last_modified
- XLSX: sheet_names, sheet_name, sheet_count, row_count, author, last_modified_by,
        created_date, last_modified
- EML: subject, sender, recipients, cc, participants, date, message_id, in_reply_to,
       references, reply_to, has_attachments, attachment_filenames
- MSG: subject, sender, recipients, cc, participants, date, message_id, in_reply_to,
       reply_to, has_attachments, attachment_filenames
"""

from __future__ import annotations

import email
import email.policy
import hashlib
import io
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedDocument:
    """파싱된 문서

    XLSX의 경우 시트별로 별도 ParsedDocument가 생성됨.
    """

    filename: str
    content: str
    file_type: str  # "pdf" | "docx" | "pptx" | "xlsx" | "txt" | "eml" | "msg"
    page_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """파일 타입에 따라 적절한 파서로 텍스트 추출

    사용법:
        parser = DocumentParser()
        result = parser.parse("path/to/document.pdf")
        # result.content, result.metadata

        # XLSX는 시트별 리스트 반환
        results = parser.parse_all("path/to/spreadsheet.xlsx")

        # 첨부파일 (바이트 데이터)
        result = parser.parse_bytes(raw_bytes, "attachment.docx")
    """

    SUPPORTED_TYPES = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".eml", ".msg"}

    def __init__(self, ocr_languages: str = "eng+kor+chi_sim") -> None:
        """DocumentParser 초기화

        Args:
            ocr_languages: Tesseract OCR 언어 설정 ('+' 구분자)
        """
        self.ocr_languages = ocr_languages

    def parse(self, file_path: str | Path) -> ParsedDocument | list[ParsedDocument]:
        """파일 타입을 감지하고 적절한 파서로 텍스트 추출

        Args:
            file_path: 파싱할 파일 경로

        Returns:
            ParsedDocument 또는 list[ParsedDocument] (XLSX 시트별)

        Raises:
            ValueError: 지원하지 않는 파일 형식
            FileNotFoundError: 파일이 존재하지 않음
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_TYPES:
            raise ValueError(f"지원하지 않는 파일 형식: {suffix}")

        logger.info(f"문서 파싱 시작: {path.name} ({suffix})")

        parser_map = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".pptx": self._parse_pptx,
            ".xlsx": self._parse_xlsx,
            ".txt": self._parse_txt,
            ".eml": self._parse_eml,
            ".msg": self._parse_msg,
        }

        result = parser_map[suffix](path)

        # 공통 메타데이터 부착
        if isinstance(result, list):
            for doc in result:
                self._attach_common_metadata(doc, path)
            logger.info(f"문서 파싱 완료: {path.name} → {len(result)}개 시트")
        else:
            self._attach_common_metadata(result, path)
            logger.info(f"문서 파싱 완료: {path.name} → {result.metadata.get('char_count', 0)}자")

        return result

    def parse_bytes(
        self, content: bytes, filename: str, source_metadata: dict[str, Any] | None = None
    ) -> ParsedDocument | list[ParsedDocument]:
        """바이트 데이터에서 텍스트 추출 (PST 첨부파일용)

        Args:
            content: 파일 바이트 데이터
            filename: 원본 파일명
            source_metadata: 첨부파일 출처 메타데이터 (이메일 발신자, 날짜 등)

        Returns:
            ParsedDocument 또는 list[ParsedDocument] (XLSX 시트별)
        """
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_TYPES:
            raise ValueError(f"지원하지 않는 파일 형식: {suffix}")

        # 메모리에서 직접 처리 가능한 타입
        memory_parsers = {
            ".pdf": self._parse_pdf_bytes,
            ".docx": self._parse_docx_bytes,
            ".pptx": self._parse_pptx_bytes,
            ".xlsx": self._parse_xlsx_bytes,
            ".txt": self._parse_txt_bytes,
            ".eml": self._parse_eml_bytes,
            ".msg": self._parse_msg_bytes,
        }

        logger.info(f"첨부파일 파싱: {filename} ({len(content)} bytes)")
        result = memory_parsers[suffix](content, filename)

        # 공통 메타데이터 부착 (바이트용)
        file_hash = hashlib.sha256(content).hexdigest()
        if isinstance(result, list):
            for doc in result:
                self._attach_common_metadata_bytes(doc, filename, len(content), file_hash)
                if source_metadata:
                    doc.metadata["source"] = source_metadata
        else:
            self._attach_common_metadata_bytes(result, filename, len(content), file_hash)
            if source_metadata:
                result.metadata["source"] = source_metadata

        return result

    # ==================== PDF ====================

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        """PDF 텍스트 추출 (PyMuPDF + Tesseract OCR 폴백)"""
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        try:
            return self._extract_pdf(doc, path.name)
        finally:
            doc.close()

    def _parse_pdf_bytes(self, content: bytes, filename: str) -> ParsedDocument:
        """PDF 바이트에서 텍스트 추출"""
        import fitz

        doc = fitz.open(stream=content, filetype="pdf")
        try:
            return self._extract_pdf(doc, filename)
        finally:
            doc.close()

    def _extract_pdf(self, doc: Any, filename: str) -> ParsedDocument:
        """PDF 문서 객체에서 텍스트 및 메타데이터 추출

        텍스트 레이어가 없거나 빈약한 페이지는 Tesseract OCR로 폴백.
        폰트 크기 분석으로 헤딩(섹션 경계)을 감지하여 sections 메타데이터에 포함.
        암호화된 PDF는 빈 본문으로 처리하고 메타데이터에 표시. 한 페이지의 텍스트
        추출이 실패해도 나머지 페이지는 계속 추출.
        """
        # 암호화 PDF 감지 — get_text가 빈 결과를 줘서 무성 실패하는 것을 방지
        is_encrypted = bool(getattr(doc, "needs_pass", False) or getattr(doc, "is_encrypted", False))
        if is_encrypted:
            logger.warning(f"PDF가 암호화되어 텍스트를 추출할 수 없음: {filename}")
            metadata: dict[str, Any] = {
                "author": "",
                "title": "",
                "page_count": len(doc),
                "is_ocr": False,
                "is_encrypted": True,
            }
            return ParsedDocument(
                filename=filename,
                content="",
                file_type="pdf",
                page_count=len(doc),
                metadata=metadata,
            )

        pages_text: list[str] = []
        ocr_used = False
        page_failures = 0
        ocr_threshold = 50  # 페이지당 최소 문자 수 — 이하이면 OCR 시도

        # 섹션 감지를 위한 블록 수집
        all_blocks: list[dict[str, Any]] = []

        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
            except Exception as e:
                page_failures += 1
                logger.debug(f"PDF 페이지 로드 실패 ({filename} p{page_num}): {e}")
                continue

            try:
                text = page.get_text("text").strip()
            except Exception as e:
                page_failures += 1
                logger.debug(f"PDF 페이지 텍스트 추출 실패 ({filename} p{page_num}): {e}")
                continue

            # 텍스트가 빈약하면 OCR 폴백
            if len(text) < ocr_threshold:
                ocr_text = self._ocr_page(page)
                if ocr_text:
                    text = ocr_text
                    ocr_used = True

            if text:
                pages_text.append(text)

            # 폰트 크기 기반 섹션 감지 (OCR 페이지 제외)
            if len(text) >= ocr_threshold and not ocr_used:
                try:
                    blocks = self._extract_text_blocks_with_font(page, page_num)
                    all_blocks.extend(blocks)
                except Exception as e:
                    logger.debug(f"PDF 섹션 감지 실패 ({filename} p{page_num}): {e}")

        if page_failures:
            logger.warning(
                f"PDF 부분 추출: {filename} — {page_failures}/{len(doc)} 페이지 추출 실패"
            )

        full_text = "\n\n".join(pages_text)

        # 섹션 경계 감지
        sections = self._detect_sections(all_blocks)

        # PDF 메타데이터 추출 (doc.metadata 자체가 None일 수도 있어서 방어)
        try:
            pdf_meta = doc.metadata or {}
        except Exception:
            pdf_meta = {}
        metadata: dict[str, Any] = {
            "author": pdf_meta.get("author", "") or "",
            "title": pdf_meta.get("title", "") or "",
            "page_count": len(doc),
            "is_ocr": ocr_used,
            "is_encrypted": False,
            # 작성자/수정자 추적용 추가 필드
            "creator": pdf_meta.get("creator", "") or "",
            "producer": pdf_meta.get("producer", "") or "",
        }
        if page_failures:
            metadata["page_failures"] = page_failures

        if sections:
            metadata["sections"] = sections

        created_date = pdf_meta.get("creationDate", "")
        if created_date:
            metadata["created_date"] = self._parse_pdf_date(created_date)

        modified_date = pdf_meta.get("modDate", "")
        if modified_date:
            metadata["modified_date"] = self._parse_pdf_date(modified_date)

        return ParsedDocument(
            filename=filename,
            content=full_text,
            file_type="pdf",
            page_count=len(doc),
            metadata=metadata,
        )

    @staticmethod
    def _extract_text_blocks_with_font(page: Any, page_num: int) -> list[dict[str, Any]]:
        """페이지에서 텍스트 블록과 폰트 크기를 추출"""
        blocks: list[dict[str, Any]] = []
        page_dict = page.get_text("dict", flags=11)  # TEXT + IMAGES flags

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # 텍스트 블록만
                continue
            for line in block.get("lines", []):
                line_text = ""
                max_font_size = 0.0
                is_bold = False
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        line_text += span_text + " "
                        font_size = span.get("size", 0.0)
                        if font_size > max_font_size:
                            max_font_size = font_size
                        font_name = span.get("font", "").lower()
                        if "bold" in font_name or (span.get("flags", 0) & 2 ** 4):
                            is_bold = True

                line_text = line_text.strip()
                if line_text:
                    blocks.append({
                        "text": line_text,
                        "font_size": max_font_size,
                        "is_bold": is_bold,
                        "page": page_num,
                    })

        return blocks

    @staticmethod
    def _detect_sections(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """폰트 크기 분석으로 헤딩(섹션 경계)을 감지

        본문의 중앙값 폰트 크기보다 1.2배 이상 큰 텍스트를 헤딩으로 판단.
        반환: [{"title": str, "page": int, "font_size": float, "offset": int}, ...]
        offset은 full_text 내에서의 대략적 문자 위치.
        """
        if not blocks:
            return []

        # 폰트 크기 중앙값 계산
        font_sizes = sorted(b["font_size"] for b in blocks if b["font_size"] > 0)
        if not font_sizes:
            return []

        median_size = font_sizes[len(font_sizes) // 2]
        heading_threshold = median_size * 1.2

        # 헤딩 감지
        sections: list[dict[str, Any]] = []
        char_offset = 0

        for block in blocks:
            text = block["text"]
            is_heading = (
                block["font_size"] >= heading_threshold
                and len(text) < 200  # 헤딩은 보통 짧음
                and (block["is_bold"] or block["font_size"] >= median_size * 1.4)
            )

            if is_heading:
                sections.append({
                    "title": text,
                    "page": block["page"],
                    "font_size": round(block["font_size"], 1),
                    "offset": char_offset,
                })

            char_offset += len(text) + 1  # +1 for newline

        return sections

    def _ocr_page(self, page: Any) -> str:
        """PyMuPDF 페이지를 이미지로 렌더링 후 Tesseract OCR 수행"""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            logger.warning("pytesseract 또는 Pillow가 설치되지 않아 OCR을 건너뜁니다.")
            return ""

        try:
            # 페이지를 이미지로 렌더링 (300 DPI)
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            text = pytesseract.image_to_string(img, lang=self.ocr_languages)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR 처리 실패: {e}")
            return ""

    @staticmethod
    def _parse_pdf_date(date_str: str) -> str:
        """PDF 날짜 문자열 파싱 (D:20240115120000+09'00' 형식)

        타임존 정보(+09'00' 등)는 제거하고 앞 14자리(yyyyMMddHHmmss)만 파싱.
        14자리 미만이면 0으로 패딩, 파싱 실패 시 원본 문자열 반환.
        """
        if not date_str:
            return ""

        # 'D:' 접두사 제거 + 숫자만 남김 (타임존 기호 제거)
        cleaned = date_str.replace("D:", "").strip()
        digits = re.sub(r"[^0-9]", "", cleaned)

        if not digits:
            return date_str

        try:
            padded = digits[:14].ljust(14, "0")
            dt = datetime.strptime(padded, "%Y%m%d%H%M%S")
            return dt.isoformat()
        except (ValueError, IndexError):
            return date_str

    # ==================== DOCX ====================

    def _parse_docx(self, path: Path) -> ParsedDocument:
        """DOCX 텍스트 추출 (python-docx)"""
        from docx import Document

        doc = Document(str(path))
        return self._extract_docx(doc, path.name)

    def _parse_docx_bytes(self, content: bytes, filename: str) -> ParsedDocument:
        """DOCX 바이트에서 텍스트 추출"""
        from docx import Document

        doc = Document(io.BytesIO(content))
        return self._extract_docx(doc, filename)

    def _extract_docx(self, doc: Any, filename: str) -> ParsedDocument:
        """DOCX 문서 객체에서 텍스트 및 메타데이터 추출

        본문 단락 + 테이블 텍스트를 포함.
        """
        # 본문 단락 추출
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        # 테이블 텍스트 추출
        table_texts: list[str] = []
        for table in doc.tables:
            for row in table.rows:
                row_text = "\t".join(cell.text.strip() for cell in row.cells)
                if row_text.strip():
                    table_texts.append(row_text)

        full_text = "\n".join(paragraphs)
        if table_texts:
            full_text += "\n\n[표]\n" + "\n".join(table_texts)

        # 메타데이터 추출
        props = doc.core_properties
        metadata: dict[str, Any] = {
            "author": props.author or "",
            "last_modified_by": props.last_modified_by or "",
        }

        if props.created:
            metadata["created_date"] = props.created.isoformat()
        if props.modified:
            metadata["last_modified"] = props.modified.isoformat()

        return ParsedDocument(
            filename=filename,
            content=full_text,
            file_type="docx",
            page_count=0,  # DOCX는 페이지 개념이 명확하지 않음
            metadata=metadata,
        )

    # ==================== PPTX ====================

    def _parse_pptx(self, path: Path) -> ParsedDocument:
        """PPTX 텍스트 추출 (python-pptx, 발표자 노트 제외)"""
        from pptx import Presentation

        prs = Presentation(str(path))
        return self._extract_pptx(prs, path.name)

    def _parse_pptx_bytes(self, content: bytes, filename: str) -> ParsedDocument:
        """PPTX 바이트에서 텍스트 추출"""
        from pptx import Presentation

        prs = Presentation(io.BytesIO(content))
        return self._extract_pptx(prs, filename)

    def _extract_pptx(self, prs: Any, filename: str) -> ParsedDocument:
        """PPTX 프레젠테이션에서 텍스트 및 메타데이터 추출

        슬라이드별 텍스트 프레임만 추출, 발표자 노트 제외.
        sections 메타데이터에 슬라이드 단위 경계 정보를 포함.
        """
        slides_text: list[str] = []
        sections: list[dict[str, Any]] = []
        char_offset = 0

        for slide_num, slide in enumerate(prs.slides, 1):
            texts: list[str] = []
            slide_title = ""

            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            texts.append(text)
                    # 슬라이드 제목 추출 (첫 번째 텍스트 또는 title placeholder)
                    if not slide_title and hasattr(shape, "placeholder_format"):
                        ph = shape.placeholder_format
                        if ph and ph.idx == 0:  # title placeholder
                            slide_title = shape.text_frame.text.strip()

                # 테이블 내 텍스트 추출
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = "\t".join(cell.text.strip() for cell in row.cells)
                        if row_text.strip():
                            texts.append(row_text)

            if texts:
                if not slide_title:
                    slide_title = texts[0][:100]  # fallback: 첫 번째 텍스트

                slide_header = f"[슬라이드 {slide_num}]"
                slide_content = f"{slide_header}\n" + "\n".join(texts)

                sections.append({
                    "title": slide_title,
                    "slide_num": slide_num,
                    "offset": char_offset,
                })

                slides_text.append(slide_content)
                char_offset += len(slide_content) + 2  # +2 for \n\n separator

        full_text = "\n\n".join(slides_text)

        # 메타데이터 추출
        props = prs.core_properties
        metadata: dict[str, Any] = {
            "author": props.author or "",
            "title": props.title or "",
            "slide_count": len(prs.slides),
            "last_modified_by": props.last_modified_by or "",
        }
        if props.created:
            metadata["created_date"] = props.created.isoformat()
        if props.modified:
            metadata["last_modified"] = props.modified.isoformat()

        if sections:
            metadata["sections"] = sections

        return ParsedDocument(
            filename=filename,
            content=full_text,
            file_type="pptx",
            page_count=len(prs.slides),
            metadata=metadata,
        )

    # ==================== XLSX ====================

    def _parse_xlsx(self, path: Path) -> list[ParsedDocument]:
        """XLSX 텍스트 추출 (openpyxl, 시트별 구분)"""
        from openpyxl import load_workbook

        wb = load_workbook(str(path), read_only=True, data_only=True)
        try:
            return self._extract_xlsx(wb, path.name)
        finally:
            wb.close()

    def _parse_xlsx_bytes(self, content: bytes, filename: str) -> list[ParsedDocument]:
        """XLSX 바이트에서 텍스트 추출"""
        from openpyxl import load_workbook

        wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        try:
            return self._extract_xlsx(wb, filename)
        finally:
            wb.close()

    def _extract_xlsx(self, wb: Any, filename: str) -> list[ParsedDocument]:
        """XLSX 워크북에서 시트별 ParsedDocument 생성

        각 시트가 독립적인 ParsedDocument로 생성됨.
        data_only=True로 수식 결과값만 추출.
        모든 시트는 동일한 워크북 작성자/수정자 메타데이터를 공유.
        """
        sheet_names = wb.sheetnames

        # 워크북 단위 작성자/수정자 정보 (시트마다 같음)
        workbook_meta: dict[str, Any] = {}
        try:
            props = wb.properties  # openpyxl DocumentProperties
            workbook_meta["author"] = props.creator or ""
            workbook_meta["last_modified_by"] = props.lastModifiedBy or ""
            if props.created:
                workbook_meta["created_date"] = props.created.isoformat()
            if props.modified:
                workbook_meta["last_modified"] = props.modified.isoformat()
            if props.title:
                workbook_meta["title"] = props.title
        except Exception as e:
            logger.debug(f"XLSX 워크북 프로퍼티 읽기 실패: {filename} ({e})")

        documents: list[ParsedDocument] = []

        for sheet_name in sheet_names:
            ws = wb[sheet_name]
            rows_text: list[str] = []
            row_count = 0

            for row in ws.iter_rows(values_only=True):
                cell_values = [str(cell) if cell is not None else "" for cell in row]
                row_text = "\t".join(cell_values)
                if row_text.strip():
                    rows_text.append(row_text)
                    row_count += 1

            content = "\n".join(rows_text)

            metadata: dict[str, Any] = {
                "sheet_name": sheet_name,
                "sheet_names": sheet_names,
                "sheet_count": len(sheet_names),
                "row_count": row_count,
                **workbook_meta,
            }

            documents.append(
                ParsedDocument(
                    filename=f"{filename} [{sheet_name}]",
                    content=content,
                    file_type="xlsx",
                    page_count=0,
                    metadata=metadata,
                )
            )

        return documents

    # ==================== TXT ====================

    def _parse_txt(self, path: Path) -> ParsedDocument:
        """TXT 텍스트 추출 (인코딩 자동 감지)"""
        # UTF-8 시도 후 실패하면 다른 인코딩 시도
        for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"):
            try:
                content = path.read_text(encoding=encoding)
                return ParsedDocument(
                    filename=path.name,
                    content=content,
                    file_type="txt",
                    metadata={},
                )
            except (UnicodeDecodeError, UnicodeError):
                continue

        raise ValueError(f"파일 인코딩을 감지할 수 없습니다: {path.name}")

    def _parse_txt_bytes(self, content: bytes, filename: str) -> ParsedDocument:
        """TXT 바이트에서 텍스트 추출"""
        for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"):
            try:
                text = content.decode(encoding)
                return ParsedDocument(
                    filename=filename,
                    content=text,
                    file_type="txt",
                    metadata={},
                )
            except (UnicodeDecodeError, UnicodeError):
                continue

        raise ValueError(f"파일 인코딩을 감지할 수 없습니다: {filename}")

    # ==================== EML ====================

    def _parse_eml(self, path: Path) -> ParsedDocument:
        """EML 이메일 파일 파싱 (RFC 5322)"""
        raw = path.read_bytes()
        return self._extract_eml(raw, path.name)

    def _parse_eml_bytes(self, content: bytes, filename: str) -> ParsedDocument:
        """EML 바이트에서 이메일 파싱"""
        return self._extract_eml(content, filename)

    def _extract_eml(self, raw: bytes, filename: str) -> ParsedDocument:
        """EML 바이트에서 본문 텍스트 및 메타데이터 추출

        text/plain 파트를 우선 사용하고, 없으면 text/html에서
        태그를 제거하여 텍스트를 추출. 손상된 MIME 구조여도 헤더만이라도
        남기도록 단계별로 방어.
        """
        # 손상된 MIME에도 가능한 정보 추출하기 위해 message_from_bytes 자체를 보호
        try:
            msg = email.message_from_bytes(raw, policy=email.policy.default)
        except Exception as e:
            logger.warning(f"EML MIME 파싱 실패 — 본문/헤더 빈 문서 반환: {filename} ({e})")
            return ParsedDocument(
                filename=filename,
                content="",
                file_type="eml",
                page_count=0,
                metadata={
                    "subject": "",
                    "sender": "",
                    "recipients": [],
                    "cc": [],
                    "participants": [],
                    "date": "",
                    "message_id": "",
                    "in_reply_to": "",
                    "references": [],
                    "reply_to": "",
                    "has_attachments": False,
                    "attachment_filenames": [],
                    "parse_error": str(e),
                },
            )

        # 본문 추출 (part-level 격리됨)
        body = self._extract_eml_body(msg)

        # 헤더 메타데이터 (안전 디코딩)
        subject = self._safe_header(msg, "Subject")
        sender = self._safe_header(msg, "From")
        date_str = self._safe_header(msg, "Date")
        message_id = self._safe_header(msg, "Message-ID")
        in_reply_to = self._safe_header(msg, "In-Reply-To")
        reply_to = self._safe_header(msg, "Reply-To")

        # References 헤더는 공백 구분된 message-id 리스트
        references_raw = self._safe_header(msg, "References")
        references = [r.strip() for r in re.split(r"\s+", references_raw) if r.strip()]

        # 수신자 파싱 (헤더 자체가 없거나 깨졌어도 빈 리스트 반환)
        try:
            recipients = self._parse_eml_addresses(msg.get_all("To"))
        except Exception as e:
            logger.warning(f"EML To 헤더 파싱 실패: {filename} ({e})")
            recipients = []
        try:
            cc = self._parse_eml_addresses(msg.get_all("Cc"))
        except Exception as e:
            logger.warning(f"EML Cc 헤더 파싱 실패: {filename} ({e})")
            cc = []

        # 날짜 파싱
        parsed_date = ""
        if date_str:
            try:
                dt = parsedate_to_datetime(date_str)
                parsed_date = dt.isoformat() if dt else ""
            except (ValueError, TypeError):
                parsed_date = date_str  # 원본 보존

        # 첨부파일 목록 수집 (walk 자체가 corrupted part에서 예외 가능)
        attachment_filenames: list[str] = []
        try:
            for part in msg.walk():
                try:
                    if part.get_content_disposition() == "attachment":
                        att_name = part.get_filename() or "unnamed"
                        attachment_filenames.append(att_name)
                except Exception as e:
                    logger.debug(f"EML part 메타 읽기 실패 (계속): {filename} ({e})")
                    continue
        except Exception as e:
            logger.warning(f"EML 첨부파일 walk 실패: {filename} ({e})")

        # 통합 참여자 (검색/UI에서 단일 리스트로 사용)
        participants = self._merge_participants(sender, recipients, cc)

        # 본문에 헤더 정보 포함 — 청크 분할 후 첫 청크는 자동으로 헤더를 포함
        # (모든 청크에 헤더를 prepend하는 작업은 chunker 단에서 수행)
        header_lines = [f"From: {sender}", f"To: {', '.join(recipients)}"]
        if cc:
            header_lines.append(f"Cc: {', '.join(cc)}")
        header_lines.append(f"Subject: {subject}")
        if parsed_date:
            header_lines.append(f"Date: {parsed_date}")
        if attachment_filenames:
            header_lines.append(f"Attachments: {', '.join(attachment_filenames)}")
        header_text = "\n".join(header_lines)

        full_text = f"{header_text}\n\n{body}"

        metadata: dict[str, Any] = {
            "subject": subject,
            "sender": sender,
            "recipients": recipients,
            "cc": cc,
            "participants": participants,
            "date": parsed_date,
            "message_id": message_id,
            "in_reply_to": in_reply_to,
            "references": references,
            "reply_to": reply_to,
            "has_attachments": len(attachment_filenames) > 0,
            "attachment_filenames": attachment_filenames,
        }

        return ParsedDocument(
            filename=filename,
            content=full_text,
            file_type="eml",
            page_count=0,
            metadata=metadata,
        )

    @staticmethod
    def _extract_eml_body(msg: Any) -> str:
        """이메일 메시지에서 본문 텍스트 추출

        text/plain 우선, 없으면 text/html에서 태그 제거. 잘못된 charset
        이나 손상된 part가 있어도 가능한 만큼 추출 (part-level 격리).
        """
        plain_parts: list[str] = []
        html_parts: list[str] = []

        try:
            walker = msg.walk()
        except Exception:
            return ""

        for part in walker:
            try:
                content_type = part.get_content_type()
                if part.get_content_disposition() == "attachment":
                    continue

                if content_type not in ("text/plain", "text/html"):
                    continue

                # 1차: policy.default 기반 디코딩
                try:
                    payload = part.get_content()
                except (LookupError, UnicodeDecodeError, AssertionError) as decode_err:
                    # 2차: raw payload + 다중 인코딩 시도
                    raw_payload = part.get_payload(decode=True)
                    if not isinstance(raw_payload, bytes):
                        logger.debug(f"EML part 디코드 실패 (skip): {decode_err}")
                        continue
                    payload = None
                    declared = (part.get_content_charset() or "").lower()
                    candidates = [declared] if declared else []
                    candidates.extend(["utf-8", "cp949", "euc-kr", "latin-1"])
                    for enc in candidates:
                        if not enc:
                            continue
                        try:
                            payload = raw_payload.decode(enc, errors="strict")
                            break
                        except (UnicodeDecodeError, LookupError):
                            continue
                    if payload is None:
                        # 마지막 fallback: 손실 허용 디코드
                        payload = raw_payload.decode("utf-8", errors="replace")

                if isinstance(payload, str):
                    if content_type == "text/plain":
                        plain_parts.append(payload)
                    else:
                        html_parts.append(payload)
            except Exception as e:
                # 한 part가 깨져도 다른 part 추출 계속
                logger.debug(f"EML part 처리 실패 (skip): {e}")
                continue

        if plain_parts:
            return "\n".join(plain_parts).strip()

        if html_parts:
            html_text = "\n".join(html_parts)
            return DocumentParser._strip_html(html_text).strip()

        return ""

    @staticmethod
    def _safe_header(msg: Any, name: str) -> str:
        """헤더를 안전하게 문자열로 디코딩 (잘못된 인코딩 방어)"""
        try:
            value = msg.get(name)
        except Exception:
            return ""
        if value is None:
            return ""
        try:
            return str(value).strip()
        except Exception:
            try:
                return repr(value)
            except Exception:
                return ""

    @staticmethod
    def _merge_participants(
        sender: str, recipients: list[str], cc: list[str]
    ) -> list[str]:
        """sender + recipients + cc를 중복 제거하여 통합 참여자 리스트 생성

        검색/UI에서 단일 필드로 노출하기 위함. 이메일 주소만 추출 (이름 부분 제외).
        """
        all_addrs: list[str] = []
        if sender:
            _, addr = parseaddr(sender)
            all_addrs.append(addr or sender)
        all_addrs.extend(recipients)
        all_addrs.extend(cc)

        seen: set[str] = set()
        unique: list[str] = []
        for a in all_addrs:
            key = a.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(a)
        return unique

    @staticmethod
    def _strip_html(html: str) -> str:
        """HTML 태그를 제거하여 텍스트 추출"""
        # <style>, <script> 블록 제거
        text = re.sub(r"<(style|script)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # <br> 태그를 줄바꿈으로
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        # <p>, <div> 태그를 줄바꿈으로
        text = re.sub(r"</(p|div|tr|li)>", "\n", text, flags=re.IGNORECASE)
        # 나머지 태그 제거
        text = re.sub(r"<[^>]+>", "", text)
        # HTML 엔티티 디코딩
        import html as html_mod
        text = html_mod.unescape(text)
        # 연속 공백/줄바꿈 정리
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _parse_eml_addresses(header_values: list[Any] | None) -> list[str]:
        """이메일 주소 헤더를 파싱하여 주소 리스트로 변환"""
        if not header_values:
            return []

        addresses: list[str] = []
        for header in header_values:
            # "Name <addr>" 또는 "addr" 형식 파싱
            raw = str(header)
            # 쉼표로 구분된 여러 주소 처리
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                _, addr = parseaddr(part)
                addresses.append(addr if addr else part)

        return [a for a in addresses if a]

    # ==================== MSG ====================

    def _parse_msg(self, path: Path) -> ParsedDocument:
        """MSG Outlook 메시지 파일 파싱 (extract-msg)"""
        import extract_msg

        with extract_msg.openMsg(str(path)) as msg:
            return self._extract_msg(msg, path.name)

    def _parse_msg_bytes(self, content: bytes, filename: str) -> ParsedDocument:
        """MSG 바이트에서 이메일 파싱"""
        import extract_msg

        with extract_msg.openMsg(content) as msg:
            return self._extract_msg(msg, filename)

    def _extract_msg(self, msg: Any, filename: str) -> ParsedDocument:
        """MSG 메시지 객체에서 본문 텍스트 및 메타데이터 추출"""
        # 본문 추출 (plain text 우선, HTML 폴백)
        body = ""
        try:
            body = msg.body or ""
        except Exception as e:
            logger.warning(f"MSG body 읽기 실패: {filename} ({e})")

        if not body or not body.strip():
            try:
                html_content = msg.htmlBody
                if html_content:
                    if isinstance(html_content, bytes):
                        html_content = html_content.decode("utf-8", errors="replace")
                    body = self._strip_html(html_content)
            except Exception as e:
                logger.debug(f"MSG htmlBody 폴백 실패: {filename} ({e})")

        subject = self._safe_attr(msg, "subject")
        sender = self._safe_attr(msg, "sender")

        date_str = ""
        try:
            raw_date = msg.date
            if raw_date:
                date_str = (
                    raw_date.isoformat() if hasattr(raw_date, "isoformat") else str(raw_date)
                )
        except (ValueError, AttributeError, Exception) as e:
            logger.debug(f"MSG date 읽기 실패: {filename} ({e})")

        message_id = self._safe_attr(msg, "messageId")
        in_reply_to = self._safe_attr(msg, "inReplyTo")
        reply_to = self._safe_attr(msg, "replyTo")

        # 수신자 파싱 (";" 또는 "," 구분 모두 허용)
        def _split_addrs(raw: str) -> list[str]:
            parts = re.split(r"[;,]", raw)
            return [p.strip() for p in parts if p.strip()]

        recipients: list[str] = []
        cc: list[str] = []
        try:
            if msg.to:
                recipients = _split_addrs(str(msg.to))
        except Exception as e:
            logger.debug(f"MSG to 헤더 파싱 실패: {filename} ({e})")
        try:
            if msg.cc:
                cc = _split_addrs(str(msg.cc))
        except Exception as e:
            logger.debug(f"MSG cc 헤더 파싱 실패: {filename} ({e})")

        # 첨부파일 목록 (각 첨부 처리 격리)
        attachment_filenames: list[str] = []
        try:
            for att in msg.attachments:
                try:
                    att_name = (
                        getattr(att, "longFilename", None)
                        or getattr(att, "shortFilename", None)
                        or "unnamed"
                    )
                    attachment_filenames.append(str(att_name))
                except Exception as e:
                    logger.debug(f"MSG 첨부파일 읽기 실패 (skip): {filename} ({e})")
                    continue
        except Exception as e:
            logger.warning(f"MSG attachments 순회 실패: {filename} ({e})")

        participants = self._merge_participants(sender, recipients, cc)

        # 본문에 헤더 정보 포함
        header_lines = [f"From: {sender}", f"To: {', '.join(recipients)}"]
        if cc:
            header_lines.append(f"Cc: {', '.join(cc)}")
        header_lines.append(f"Subject: {subject}")
        if date_str:
            header_lines.append(f"Date: {date_str}")
        if attachment_filenames:
            header_lines.append(f"Attachments: {', '.join(attachment_filenames)}")
        header_text = "\n".join(header_lines)

        full_text = f"{header_text}\n\n{body}"

        metadata: dict[str, Any] = {
            "subject": subject,
            "sender": sender,
            "recipients": recipients,
            "cc": cc,
            "participants": participants,
            "date": date_str,
            "message_id": message_id,
            "in_reply_to": in_reply_to,
            "reply_to": reply_to,
            "has_attachments": len(attachment_filenames) > 0,
            "attachment_filenames": attachment_filenames,
        }

        return ParsedDocument(
            filename=filename,
            content=full_text,
            file_type="msg",
            page_count=0,
            metadata=metadata,
        )

    @staticmethod
    def _safe_attr(obj: Any, name: str) -> str:
        """객체 속성을 안전하게 문자열로 가져오기 (예외 시 빈 문자열)"""
        try:
            value = getattr(obj, name, None)
        except Exception:
            return ""
        if value is None:
            return ""
        try:
            return str(value).strip()
        except Exception:
            return ""

    # ==================== 공통 메타데이터 ====================

    def _attach_common_metadata(self, doc: ParsedDocument, path: Path) -> None:
        """파일 기반 공통 메타데이터 부착"""
        stat = path.stat()

        doc.metadata.update({
            "filename": path.name,
            "file_path": str(path),
            "file_size": stat.st_size,
            "file_type": doc.file_type,
            "file_hash": self._compute_file_hash(path),
            "char_count": len(doc.content),
            "language": self._detect_language(doc.content),
        })

    def _attach_common_metadata_bytes(
        self, doc: ParsedDocument, filename: str, file_size: int, file_hash: str
    ) -> None:
        """바이트 기반 공통 메타데이터 부착"""
        doc.metadata.update({
            "filename": filename,
            "file_path": "",
            "file_size": file_size,
            "file_type": doc.file_type,
            "file_hash": file_hash,
            "char_count": len(doc.content),
            "language": self._detect_language(doc.content),
        })

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """파일 SHA-256 해시 계산"""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _detect_language(text: str) -> str:
        """텍스트 언어 감지

        langdetect 사용. 감지 실패 시 'unknown' 반환.
        """
        if not text or len(text.strip()) < 20:
            return "unknown"

        try:
            from langdetect import detect

            return detect(text)
        except Exception:
            return "unknown"
