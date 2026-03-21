"""PST 파일 파서 — 이메일 및 Teams 채팅 PST 처리

지원 라이브러리 우선순위:
1. pypff (libpff) — 가장 직접적인 PST 접근
2. libratom — pypff 래퍼, 설치 편의성
3. Mock 모드 — 라이브러리 없이 개발/테스트

사용법:
    parser = PSTParser("path/to/file.pst")
    result = parser.parse()
    # result.emails, result.chats, result.attachments
"""

from __future__ import annotations

import email
import email.policy
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# === 라이브러리 감지 ===
PST_BACKEND = "none"

try:
    import pypff

    PST_BACKEND = "pypff"
except ImportError:
    try:
        from libratom.lib.pff import PffArchive

        PST_BACKEND = "libratom"
    except ImportError:
        pass


# === 데이터 모델 ===


@dataclass
class Attachment:
    """이메일/채팅 첨부파일"""

    filename: str
    content: bytes
    mime_type: str = ""
    size: int = 0

    def __post_init__(self) -> None:
        if not self.size:
            self.size = len(self.content)


@dataclass
class EmailMessage:
    """파싱된 이메일 메시지"""

    subject: str
    sender: str
    recipients: list[str]
    body: str
    date: datetime | None
    attachments: list[Attachment] = field(default_factory=list)
    message_id: str = ""
    folder_path: str = ""
    html_body: str = ""

    @property
    def has_attachments(self) -> bool:
        return len(self.attachments) > 0


@dataclass
class ChatMessage:
    """파싱된 Teams 채팅 메시지"""

    sender: str
    body: str
    timestamp: datetime | None
    conversation_id: str = ""
    chat_type: str = "1:1"
    participants: list[str] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)


@dataclass
class PSTParseResult:
    """PST 파싱 결과"""

    emails: list[EmailMessage] = field(default_factory=list)
    chats: list[ChatMessage] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)
    total_messages: int = 0
    total_folders: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def summary(self) -> dict:
        return {
            "total_messages": self.total_messages,
            "emails": len(self.emails),
            "chats": len(self.chats),
            "attachments": len(self.attachments),
            "total_folders": self.total_folders,
            "errors": len(self.errors),
        }


# === 파서 구현 ===


class PSTParser:
    """PST 파일을 파싱하여 이메일, 채팅, 첨부파일을 추출"""

    TEAMS_FOLDER_PATTERNS = [
        "Teams Chat",
        "Conversation History",
        "Team Chat",
        "TeamsMessagesData",
    ]

    def __init__(self, pst_path: str | Path) -> None:
        self.pst_path = Path(pst_path)
        if not self.pst_path.exists():
            raise FileNotFoundError(f"PST 파일을 찾을 수 없습니다: {self.pst_path}")
        if self.pst_path.suffix.lower() not in (".pst", ".ost"):
            raise ValueError(f"PST/OST 파일이 아닙니다: {self.pst_path.suffix}")

        self._backend = PST_BACKEND
        logger.info(f"PSTParser 초기화: {self.pst_path.name} (백엔드: {self._backend})")

    def parse(self) -> PSTParseResult:
        """PST 파일 전체 파싱"""
        logger.info(f"PST 파싱 시작: {self.pst_path.name}")

        if self._backend == "pypff":
            result = self._parse_with_pypff()
        elif self._backend == "libratom":
            result = self._parse_with_libratom()
        else:
            raise RuntimeError(
                "PST 파싱 라이브러리가 설치되지 않았습니다. "
                "pip install libpff-python 또는 pip install libratom 을 실행하세요. "
                "개발/테스트 시에는 MockPSTParser를 사용하세요."
            )

        logger.info(f"PST 파싱 완료: {result.summary}")
        return result

    # ─── pypff 백엔드 ───

    def _parse_with_pypff(self) -> PSTParseResult:
        """pypff(libpff)를 사용한 PST 파싱"""
        result = PSTParseResult()

        try:
            pff_file = pypff.file()
            pff_file.open(str(self.pst_path))
            root = pff_file.get_root_folder()

            self._walk_pypff_folder(root, "", result)

            pff_file.close()
        except Exception as e:
            error_msg = f"pypff 파싱 오류: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        return result

    def _walk_pypff_folder(
        self,
        folder: Any,
        path: str,
        result: PSTParseResult,
    ) -> None:
        """pypff 폴더를 재귀적으로 순회"""
        folder_name = _safe_str(folder.name) or "(root)"
        current_path = f"{path}/{folder_name}" if path else folder_name
        result.total_folders += 1

        is_teams = self._is_teams_folder(folder_name)

        # 메시지 처리
        for i in range(folder.number_of_sub_messages):
            try:
                message = folder.get_sub_message(i)
                result.total_messages += 1

                if is_teams:
                    chat = self._pypff_to_chat(message, current_path)
                    if chat:
                        result.chats.append(chat)
                        result.attachments.extend(chat.attachments)
                else:
                    email_msg = self._pypff_to_email(message, current_path)
                    if email_msg:
                        result.emails.append(email_msg)
                        result.attachments.extend(email_msg.attachments)
            except Exception as e:
                error_msg = f"메시지 파싱 오류 [{current_path}#{i}]: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)

        # 하위 폴더 재귀
        for j in range(folder.number_of_sub_folders):
            sub_folder = folder.get_sub_folder(j)
            self._walk_pypff_folder(sub_folder, current_path, result)

    def _pypff_to_email(self, message: Any, folder_path: str) -> EmailMessage | None:
        """pypff 메시지 → EmailMessage"""
        try:
            subject = _safe_str(message.subject)
            sender = _safe_str(message.sender_name)
            body = _safe_str(message.plain_text_body)
            html_body = _safe_str(message.html_body)
            if not body and html_body:
                body = _strip_html(html_body)

            date = _extract_pypff_date(message)
            recipients = _extract_pypff_recipients(message)
            attachments = _extract_pypff_attachments(message)

            return EmailMessage(
                subject=subject,
                sender=sender,
                recipients=recipients,
                body=body,
                html_body=html_body,
                date=date,
                attachments=attachments,
                message_id=str(message.identifier),
                folder_path=folder_path,
            )
        except Exception as e:
            logger.warning(f"이메일 변환 오류: {e}")
            return None

    def _pypff_to_chat(self, message: Any, folder_path: str) -> ChatMessage | None:
        """pypff 메시지 → ChatMessage"""
        try:
            sender = _safe_str(message.sender_name)
            body = _safe_str(message.plain_text_body)
            if not body:
                html = _safe_str(message.html_body)
                body = _strip_html(html) if html else ""

            timestamp = _extract_pypff_date(message)
            recipients = _extract_pypff_recipients(message)
            participants = list(set(p for p in [sender] + recipients if p))
            chat_type = "group" if len(participants) > 2 else "1:1"
            conv_id = f"{folder_path}__{'_'.join(sorted(participants))}"
            attachments = _extract_pypff_attachments(message)

            return ChatMessage(
                sender=sender,
                body=body,
                timestamp=timestamp,
                conversation_id=conv_id,
                chat_type=chat_type,
                participants=participants,
                attachments=attachments,
            )
        except Exception as e:
            logger.warning(f"채팅 변환 오류: {e}")
            return None

    # ─── libratom 백엔드 ───

    def _parse_with_libratom(self) -> PSTParseResult:
        """libratom을 사용한 PST 파싱"""
        result = PSTParseResult()

        try:
            archive = PffArchive(str(self.pst_path))

            for folder in archive.folders():
                result.total_folders += 1
                folder_name = folder.name or "(unknown)"
                is_teams = self._is_teams_folder(folder_name)

                for j in range(folder.get_number_of_sub_messages()):
                    try:
                        pff_msg = folder.get_sub_message(j)
                        result.total_messages += 1

                        eml_str = archive.format_message(pff_msg)
                        parsed = email.message_from_string(
                            eml_str, policy=email.policy.default
                        )

                        if is_teams:
                            chat = _eml_to_chat(parsed, folder_name)
                            if chat:
                                result.chats.append(chat)
                                result.attachments.extend(chat.attachments)
                        else:
                            email_msg = _eml_to_email(parsed, folder_name)
                            if email_msg:
                                result.emails.append(email_msg)
                                result.attachments.extend(email_msg.attachments)
                    except Exception as e:
                        result.errors.append(f"libratom 메시지 파싱 오류: {e}")

        except Exception as e:
            result.errors.append(f"libratom 파싱 오류: {e}")

        return result

    # ─── 공통 ───

    def _is_teams_folder(self, folder_name: str) -> bool:
        """폴더명이 Teams 채팅 관련인지 판별"""
        name_lower = folder_name.lower()
        return any(p.lower() in name_lower for p in self.TEAMS_FOLDER_PATTERNS)


# === 헬퍼 함수 (모듈 레벨) ===


def _safe_str(value: Any) -> str:
    """None-safe 문자열 변환"""
    if value is None:
        return ""
    try:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
    except Exception:
        return ""


def _strip_html(html: str) -> str:
    """HTML 태그를 제거하고 텍스트만 추출"""
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pypff_date(message: Any) -> datetime | None:
    """pypff 메시지에서 날짜 추출"""
    for attr in ("delivery_time", "client_submit_time", "creation_time"):
        try:
            dt = getattr(message, attr, None)
            if dt:
                return dt
        except Exception:
            continue
    return None


def _extract_pypff_recipients(message: Any) -> list[str]:
    """pypff 메시지에서 수신자 목록 추출"""
    recipients = []
    try:
        for i in range(message.number_of_recipients):
            recipient = message.get_recipient(i)
            name = _safe_str(recipient.display_name)
            if name:
                recipients.append(name)
    except Exception:
        try:
            headers = _safe_str(message.transport_headers)
            if headers:
                for line in headers.split("\n"):
                    if line.lower().startswith("to:"):
                        recipients.append(line[3:].strip())
        except Exception:
            pass
    return recipients


def _extract_pypff_attachments(message: Any) -> list[Attachment]:
    """pypff 메시지에서 첨부파일 추출"""
    attachments = []
    try:
        for i in range(message.number_of_attachments):
            try:
                att = message.get_attachment(i)
                filename = _safe_str(att.name) or f"attachment_{i}"
                size = att.get_size()
                content = att.read_buffer(size) if size > 0 else b""
                attachments.append(
                    Attachment(filename=filename, content=content, size=size)
                )
            except Exception as e:
                logger.warning(f"첨부파일 추출 오류 [{i}]: {e}")
    except Exception:
        pass
    return attachments


def _eml_to_email(msg: email.message.Message, folder: str) -> EmailMessage | None:
    """파싱된 eml → EmailMessage"""
    try:
        subject = str(msg.get("subject", ""))
        sender = str(msg.get("from", ""))
        recipients = [r.strip() for r in str(msg.get("to", "")).split(",") if r.strip()]
        message_id = str(msg.get("message-id", ""))

        body, html_body = "", ""
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == "text/plain" and not body:
                    body = part.get_content()
                elif ct == "text/html" and not html_body:
                    html_body = part.get_content()
        else:
            content = msg.get_content()
            if msg.get_content_type() == "text/plain":
                body = content
            else:
                html_body = content
                body = _strip_html(content)

        date = None
        if date_str := msg.get("date", ""):
            try:
                from email.utils import parsedate_to_datetime
                date = parsedate_to_datetime(date_str)
            except Exception:
                pass

        attachments = _extract_eml_attachments(msg)

        return EmailMessage(
            subject=subject, sender=sender, recipients=recipients,
            body=body, html_body=html_body, date=date,
            attachments=attachments, message_id=message_id, folder_path=folder,
        )
    except Exception as e:
        logger.warning(f"eml 이메일 변환 오류: {e}")
        return None


def _eml_to_chat(msg: email.message.Message, folder: str) -> ChatMessage | None:
    """파싱된 eml → ChatMessage"""
    try:
        sender = str(msg.get("from", ""))
        recipients = [r.strip() for r in str(msg.get("to", "")).split(",") if r.strip()]

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_content()
                    break
        else:
            if msg.get_content_type() == "text/plain":
                body = msg.get_content()

        date = None
        if date_str := msg.get("date", ""):
            try:
                from email.utils import parsedate_to_datetime
                date = parsedate_to_datetime(date_str)
            except Exception:
                pass

        participants = list(set([sender] + recipients))
        chat_type = "group" if len(participants) > 2 else "1:1"
        conv_id = f"{folder}__{'_'.join(sorted(participants))}"
        attachments = _extract_eml_attachments(msg)

        return ChatMessage(
            sender=sender, body=body, timestamp=date,
            conversation_id=conv_id, chat_type=chat_type,
            participants=participants, attachments=attachments,
        )
    except Exception as e:
        logger.warning(f"eml 채팅 변환 오류: {e}")
        return None


def _extract_eml_attachments(msg: email.message.Message) -> list[Attachment]:
    """eml 메시지에서 첨부파일 추출"""
    attachments = []
    if not msg.is_multipart():
        return attachments

    for part in msg.walk():
        disposition = part.get("Content-Disposition", "")
        if "attachment" in disposition:
            filename = part.get_filename() or "unknown_attachment"
            content = part.get_payload(decode=True) or b""
            attachments.append(
                Attachment(
                    filename=filename,
                    content=content,
                    mime_type=part.get_content_type(),
                )
            )
    return attachments


# === Mock 파서 (개발/테스트용) ===


class MockPSTParser:
    """PST 라이브러리 없이 개발/테스트 가능한 Mock 파서

    사용법:
        parser = MockPSTParser()
        result = parser.parse()
    """

    def parse(self) -> PSTParseResult:
        """샘플 데이터가 포함된 파싱 결과 반환"""
        logger.info("Mock PST 파싱 — 샘플 데이터 생성")
        return PSTParseResult(
            emails=[
                EmailMessage(
                    subject="[내부] Q1 감사 보고서 검토 요청",
                    sender="김감사",
                    recipients=["박대리", "이과장"],
                    body="첨부된 Q1 감사 보고서를 검토 부탁드립니다.\n"
                    "특히 매출 인식 부분 집중 검토 바랍니다.",
                    date=datetime(2026, 3, 14, 9, 30),
                    folder_path="받은 편지함",
                    message_id="mock_001",
                    attachments=[
                        Attachment(
                            filename="Q1_감사보고서_v2.pdf",
                            content=b"<mock pdf>",
                            mime_type="application/pdf",
                        )
                    ],
                ),
                EmailMessage(
                    subject="Re: [내부] Q1 감사 보고서 검토 요청",
                    sender="박대리",
                    recipients=["김감사"],
                    body="확인했습니다. 매출 인식 기준 변경 사항이 있어서\n"
                    "별도 의견서를 첨부합니다.",
                    date=datetime(2026, 3, 14, 14, 15),
                    folder_path="받은 편지함",
                    message_id="mock_002",
                    attachments=[
                        Attachment(
                            filename="매출인식_검토의견.docx",
                            content=b"<mock docx>",
                            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        )
                    ],
                ),
            ],
            chats=[
                ChatMessage(
                    sender="김감사",
                    body="박대리님, 견적서 확인하셨나요?",
                    timestamp=datetime(2026, 3, 14, 10, 0),
                    conversation_id="chat_김감사_박대리_20260314",
                    chat_type="1:1",
                    participants=["김감사", "박대리"],
                ),
                ChatMessage(
                    sender="박대리",
                    body="네, 단가가 좀 높은데 15% 할인 가능할까요?",
                    timestamp=datetime(2026, 3, 14, 10, 5),
                    conversation_id="chat_김감사_박대리_20260314",
                    chat_type="1:1",
                    participants=["김감사", "박대리"],
                ),
                ChatMessage(
                    sender="김감사",
                    body="할인 적용해서 수정 견적 보내드리겠습니다.",
                    timestamp=datetime(2026, 3, 14, 10, 12),
                    conversation_id="chat_김감사_박대리_20260314",
                    chat_type="1:1",
                    participants=["김감사", "박대리"],
                ),
            ],
            total_messages=5,
            total_folders=3,
        )
