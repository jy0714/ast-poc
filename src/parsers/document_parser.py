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
- PDF: author, created_date, page_count, title, is_ocr
- DOCX: author, created_date, last_modified, last_modified_by
- PPTX: author, slide_count, title
- XLSX: sheet_names, sheet_name, sheet_count, row_count
- EML: subject, sender, recipients, cc, date, message_id, has_attachments, attachment_filenames
- MSG: subject, sender, recipients, cc, date, message_id, has_attachments, attachment_filenames
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
        """
        import fitz

        pages_text: list[str] = []
        ocr_used = False
        ocr_threshold = 50  # 페이지당 최소 문자 수 — 이하이면 OCR 시도

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            # 텍스트가 빈약하면 OCR 폴백
            if len(text) < ocr_threshold:
                ocr_text = self._ocr_page(page)
                if ocr_text:
                    text = ocr_text
                    ocr_used = True

            if text:
                pages_text.append(text)

        full_text = "\n\n".join(pages_text)

        # PDF 메타데이터 추출
        pdf_meta = doc.metadata or {}
        metadata: dict[str, Any] = {
            "author": pdf_meta.get("author", ""),
            "title": pdf_meta.get("title", ""),
            "page_count": len(doc),
            "is_ocr": ocr_used,
        }

        created_date = pdf_meta.get("creationDate", "")
        if created_date:
            metadata["created_date"] = self._parse_pdf_date(created_date)

        return ParsedDocument(
            filename=filename,
            content=full_text,
            file_type="pdf",
            page_count=len(doc),
            metadata=metadata,
        )

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
        """PDF 날짜 문자열 파싱 (D:20240115120000+09'00' 형식)"""
        if not date_str:
            return ""

        # 'D:' 접두사 제거
        cleaned = date_str.replace("D:", "").strip()

        # 타임존 정보 제거 후 기본 파싱
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d"):
            try:
                # 타임존 부분(+09'00' 등) 제거
                date_part = cleaned[:len(fmt.replace("%", "").replace("Y", "YYYY").replace("m", "MM").replace("d", "DD").replace("H", "HH").replace("M", "MM").replace("S", "SS"))]
                dt = datetime.strptime(cleaned[:14].ljust(14, "0"), "%Y%m%d%H%M%S")
                return dt.isoformat()
            except (ValueError, IndexError):
                continue

        return date_str  # 파싱 실패 시 원본 반환

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
        """
        slides_text: list[str] = []

        for slide_num, slide in enumerate(prs.slides, 1):
            texts: list[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            texts.append(text)

                # 테이블 내 텍스트 추출
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = "\t".join(cell.text.strip() for cell in row.cells)
                        if row_text.strip():
                            texts.append(row_text)

            if texts:
                slide_header = f"[슬라이드 {slide_num}]"
                slides_text.append(f"{slide_header}\n" + "\n".join(texts))

        full_text = "\n\n".join(slides_text)

        # 메타데이터 추출
        props = prs.core_properties
        metadata: dict[str, Any] = {
            "author": props.author or "",
            "title": props.title or "",
            "slide_count": len(prs.slides),
        }

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
        """
        sheet_names = wb.sheetnames
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
        태그를 제거하여 텍스트를 추출.
        """
        msg = email.message_from_bytes(raw, policy=email.policy.default)

        # 본문 추출
        body = self._extract_eml_body(msg)

        # 헤더 메타데이터
        subject = str(msg.get("Subject", ""))
        sender = str(msg.get("From", ""))
        date_str = msg.get("Date", "")
        message_id = str(msg.get("Message-ID", ""))

        # 수신자 파싱
        recipients = self._parse_eml_addresses(msg.get_all("To"))
        cc = self._parse_eml_addresses(msg.get_all("Cc"))

        # 날짜 파싱
        parsed_date = ""
        if date_str:
            try:
                dt = parsedate_to_datetime(str(date_str))
                parsed_date = dt.isoformat()
            except (ValueError, TypeError):
                parsed_date = str(date_str)

        # 첨부파일 목록 수집
        attachment_filenames: list[str] = []
        for part in msg.walk():
            disposition = part.get_content_disposition()
            if disposition == "attachment":
                att_name = part.get_filename() or "unnamed"
                attachment_filenames.append(att_name)

        # 본문에 헤더 정보 포함하여 검색 가능하게 구성
        header_text = f"From: {sender}\nTo: {', '.join(recipients)}"
        if cc:
            header_text += f"\nCc: {', '.join(cc)}"
        header_text += f"\nSubject: {subject}"
        if parsed_date:
            header_text += f"\nDate: {parsed_date}"

        full_text = f"{header_text}\n\n{body}"

        metadata: dict[str, Any] = {
            "subject": subject,
            "sender": sender,
            "recipients": recipients,
            "cc": cc,
            "date": parsed_date,
            "message_id": message_id,
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

        text/plain 우선, 없으면 text/html에서 태그 제거.
        """
        plain_parts: list[str] = []
        html_parts: list[str] = []

        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = part.get_content_disposition()

            # 첨부파일 파트는 건너뛰기
            if disposition == "attachment":
                continue

            if content_type == "text/plain":
                payload = part.get_content()
                if isinstance(payload, str):
                    plain_parts.append(payload)
            elif content_type == "text/html":
                payload = part.get_content()
                if isinstance(payload, str):
                    html_parts.append(payload)

        if plain_parts:
            return "\n".join(plain_parts).strip()

        if html_parts:
            # HTML 태그 제거
            html_text = "\n".join(html_parts)
            return DocumentParser._strip_html(html_text).strip()

        return ""

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
        body = msg.body or ""
        if not body.strip() and msg.htmlBody:
            html_content = msg.htmlBody
            if isinstance(html_content, bytes):
                html_content = html_content.decode("utf-8", errors="replace")
            body = self._strip_html(html_content)

        subject = msg.subject or ""
        sender = msg.sender or ""
        date_str = ""
        if msg.date:
            try:
                date_str = msg.date.isoformat() if hasattr(msg.date, "isoformat") else str(msg.date)
            except (ValueError, AttributeError):
                date_str = str(msg.date)

        message_id = msg.messageId or ""

        # 수신자 파싱
        recipients: list[str] = []
        cc: list[str] = []
        if msg.to:
            recipients = [addr.strip() for addr in str(msg.to).split(";") if addr.strip()]
        if msg.cc:
            cc = [addr.strip() for addr in str(msg.cc).split(";") if addr.strip()]

        # 첨부파일 목록
        attachment_filenames: list[str] = []
        for att in msg.attachments:
            att_name = getattr(att, "longFilename", None) or getattr(att, "shortFilename", None) or "unnamed"
            attachment_filenames.append(att_name)

        # 본문에 헤더 정보 포함
        header_text = f"From: {sender}\nTo: {', '.join(recipients)}"
        if cc:
            header_text += f"\nCc: {', '.join(cc)}"
        header_text += f"\nSubject: {subject}"
        if date_str:
            header_text += f"\nDate: {date_str}"

        full_text = f"{header_text}\n\n{body}"

        metadata: dict[str, Any] = {
            "subject": subject,
            "sender": sender,
            "recipients": recipients,
            "cc": cc,
            "date": date_str,
            "message_id": message_id,
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
