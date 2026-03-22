"""스마트 청킹 엔진 — 소스 타입별 최적 분할

4종 청커:
- DocumentChunker: 문서(PDF/DOCX/PPTX/XLSX/TXT/EML/MSG) — RecursiveCharacterTextSplitter 기반
- ChatChunker: Teams 채팅 — 시간 윈도우 기반, 대화 맥락 보존
- EmailChunker: 이메일 — 스레드 기반 그룹핑, 긴 스레드 분할
- AttachmentChunker: 첨부파일 — DocumentParser로 텍스트 추출 후 DocumentChunker 위임

각 청커는 source_type과 메타데이터를 청크에 부착하여
검색 시 소스 타입별 필터링과 출처 추적이 가능하도록 함.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# === 데이터 모델 ===


@dataclass
class Chunk:
    """청킹된 텍스트 단위

    Attributes:
        content: 청크 텍스트
        metadata: 검색/필터링용 메타데이터
        chunk_id: 고유 식별자 (결정적 해시)
        source_type: 소스 유형 ("email" | "teams_chat" | "document" | "attachment")
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    source_type: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = self._generate_id()

    def _generate_id(self) -> str:
        """콘텐츠 + 메타데이터 기반 결정적 ID 생성"""
        source = f"{self.source_type}:{self.content[:200]}:{sorted(self.metadata.items())}"
        return hashlib.md5(source.encode("utf-8")).hexdigest()[:16]


# === DocumentChunker ===


class DocumentChunker:
    """문서용 청커 — LangChain RecursiveCharacterTextSplitter 기반

    문서 텍스트를 의미 단위(단락, 문장, 단어)를 존중하면서
    설정된 크기로 분할. 오버랩으로 맥락 연속성 보장.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size_docs
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap_docs

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """문서 텍스트를 청크로 분할

        Args:
            text: 분할할 텍스트
            metadata: 모든 청크에 부착할 공통 메타데이터

        Returns:
            Chunk 리스트 (빈 텍스트이면 빈 리스트)
        """
        if not text or not text.strip():
            return []

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=len,
        )

        splits = splitter.split_text(text)
        base_meta = metadata or {}

        chunks: list[Chunk] = []
        for i, split_text in enumerate(splits):
            chunk_meta = {
                **base_meta,
                "chunk_index": i,
                "total_chunks": len(splits),
            }
            chunks.append(
                Chunk(
                    content=split_text,
                    metadata=chunk_meta,
                    source_type="document",
                )
            )

        logger.info(
            f"문서 청킹 완료: {len(text)}자 → {len(chunks)}개 청크 "
            f"(크기={self.chunk_size}, 오버랩={self.chunk_overlap})"
        )
        return chunks

    def chunk_parsed_document(
        self, parsed_doc: Any, extra_metadata: dict[str, Any] | None = None
    ) -> list[Chunk]:
        """ParsedDocument 객체를 직접 청킹

        ParsedDocument의 메타데이터를 각 청크에 전파.

        Args:
            parsed_doc: DocumentParser가 반환한 ParsedDocument
            extra_metadata: 추가 메타데이터 (case_id 등)
        """
        meta = {**parsed_doc.metadata}
        if extra_metadata:
            meta.update(extra_metadata)

        return self.chunk(parsed_doc.content, metadata=meta)


# === ChatChunker ===


class ChatChunker:
    """채팅용 청커 — 시간 윈도우 + 대화 ID 기반 분할

    동일 conversation_id 내에서 시간 간격이 window_minutes를 초과하면
    새로운 대화 블록으로 분할. 각 블록이 하나의 청크가 됨.
    """

    def __init__(self, window_minutes: int | None = None) -> None:
        self.window_minutes = window_minutes or settings.chat_window_minutes

    def chunk(
        self, messages: list[dict[str, Any]], metadata: dict[str, Any] | None = None
    ) -> list[Chunk]:
        """채팅 메시지를 시간 윈도우 기준으로 대화 블록으로 분할

        Args:
            messages: [{"sender": str, "body": str, "timestamp": datetime}, ...]
                      timestamp 기준 오름차순 정렬 권장
            metadata: 공통 메타데이터 (participants, chat_type, conversation_id 등)

        Returns:
            Chunk 리스트 (각 청크 = 하나의 대화 블록)
        """
        if not messages:
            return []

        # timestamp 기준 정렬
        sorted_msgs = sorted(
            messages,
            key=lambda m: m.get("timestamp") or datetime.min,
        )

        # 시간 윈도우 기반 그룹핑
        groups: list[list[dict[str, Any]]] = []
        current_group: list[dict[str, Any]] = [sorted_msgs[0]]

        for msg in sorted_msgs[1:]:
            prev_ts = current_group[-1].get("timestamp")
            curr_ts = msg.get("timestamp")

            if prev_ts and curr_ts:
                gap = (curr_ts - prev_ts).total_seconds() / 60
                if gap > self.window_minutes:
                    groups.append(current_group)
                    current_group = []

            current_group.append(msg)

        if current_group:
            groups.append(current_group)

        # 각 그룹을 청크로 변환
        base_meta = metadata or {}
        chunks: list[Chunk] = []

        for group_idx, group in enumerate(groups):
            text = self._format_chat_block(group)

            # 대화 블록 메타데이터
            timestamps = [m["timestamp"] for m in group if m.get("timestamp")]
            participants = list({m["sender"] for m in group if m.get("sender")})

            chunk_meta = {
                **base_meta,
                "chunk_index": group_idx,
                "total_chunks": len(groups),
                "message_count": len(group),
                "participants": participants,
            }

            if timestamps:
                chunk_meta["date_range_start"] = min(timestamps).isoformat()
                chunk_meta["date_range_end"] = max(timestamps).isoformat()

            chunks.append(
                Chunk(
                    content=text,
                    metadata=chunk_meta,
                    source_type="teams_chat",
                )
            )

        logger.info(
            f"채팅 청킹 완료: {len(messages)}개 메시지 → {len(chunks)}개 블록 "
            f"(윈도우={self.window_minutes}분)"
        )
        return chunks

    def chunk_chat_messages(
        self,
        chat_messages: list[Any],
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """PSTParser의 ChatMessage 객체 리스트를 직접 청킹

        conversation_id별로 그룹핑 후 각 그룹을 시간 윈도우로 분할.
        """
        # conversation_id별 그룹핑
        by_conv: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chat in chat_messages:
            conv_id = getattr(chat, "conversation_id", "unknown")
            by_conv[conv_id].append({
                "sender": chat.sender,
                "body": chat.body,
                "timestamp": chat.timestamp,
            })

        all_chunks: list[Chunk] = []
        for conv_id, msgs in by_conv.items():
            meta = {
                "conversation_id": conv_id,
                "chat_type": getattr(chat_messages[0], "chat_type", "unknown"),
            }
            if extra_metadata:
                meta.update(extra_metadata)

            chunks = self.chunk(msgs, metadata=meta)
            all_chunks.extend(chunks)

        return all_chunks

    @staticmethod
    def _format_chat_block(messages: list[dict[str, Any]]) -> str:
        """채팅 메시지 그룹을 읽기 쉬운 텍스트로 포맷"""
        lines: list[str] = []
        for msg in messages:
            sender = msg.get("sender", "알 수 없음")
            body = msg.get("body", "")
            ts = msg.get("timestamp")
            time_str = ts.strftime("%H:%M") if ts else ""

            if time_str:
                lines.append(f"[{time_str}] {sender}: {body}")
            else:
                lines.append(f"{sender}: {body}")

        return "\n".join(lines)


# === EmailChunker ===


class EmailChunker:
    """이메일용 청커 — 스레드 기반 그룹핑

    동일 스레드(Re:/Fwd: 제거 후 subject 매칭)의 이메일을
    시간순으로 합쳐 하나의 청크로 생성. 스레드가 max_chars를
    초과하면 DocumentChunker로 2차 분할.
    """

    # Re: / Fwd: / RE: / FW: 등 접두사 패턴
    _THREAD_PREFIX_RE = re.compile(
        r"^(?:(?:re|fw|fwd)\s*:\s*)+",
        re.IGNORECASE,
    )

    def __init__(self, max_chars: int | None = None) -> None:
        self.max_chars = max_chars or settings.email_thread_max_chars

    def chunk(
        self,
        emails: list[Any],
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """이메일 리스트를 스레드 기반으로 청킹

        Args:
            emails: EmailMessage 객체 리스트 (PSTParser 결과)
            metadata: 공통 메타데이터 (case_id 등)

        Returns:
            Chunk 리스트
        """
        if not emails:
            return []

        # 스레드 그룹핑 (정규화된 subject 기준)
        threads: dict[str, list[Any]] = defaultdict(list)
        for email_msg in emails:
            thread_key = self._normalize_subject(email_msg.subject)
            threads[thread_key].append(email_msg)

        # 각 스레드를 시간순 정렬 후 청크 생성
        base_meta = metadata or {}
        all_chunks: list[Chunk] = []

        for thread_subject, thread_emails in threads.items():
            # 시간순 정렬
            sorted_emails = sorted(
                thread_emails,
                key=lambda e: e.date or datetime.min,
            )

            thread_text = self._format_thread(sorted_emails)

            # 스레드 메타데이터
            participants = list({
                p
                for e in sorted_emails
                for p in [e.sender] + e.recipients
                if p
            })
            dates = [e.date for e in sorted_emails if e.date]

            thread_meta: dict[str, Any] = {
                **base_meta,
                "thread_subject": thread_subject,
                "email_count": len(sorted_emails),
                "participants": participants,
                "has_attachments": any(e.has_attachments for e in sorted_emails),
            }

            if dates:
                thread_meta["date_range_start"] = min(dates).isoformat()
                thread_meta["date_range_end"] = max(dates).isoformat()

            # 스레드가 max_chars 초과 시 2차 분할
            if len(thread_text) > self.max_chars:
                doc_chunker = DocumentChunker(
                    chunk_size=self.max_chars,
                    chunk_overlap=min(200, self.max_chars // 5),
                )
                sub_chunks = doc_chunker.chunk(thread_text, metadata=thread_meta)
                # source_type을 email로 덮어쓰기
                for sc in sub_chunks:
                    sc.source_type = "email"
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(
                    Chunk(
                        content=thread_text,
                        metadata=thread_meta,
                        source_type="email",
                    )
                )

        logger.info(
            f"이메일 청킹 완료: {len(emails)}개 이메일, {len(threads)}개 스레드 "
            f"→ {len(all_chunks)}개 청크"
        )
        return all_chunks

    def _normalize_subject(self, subject: str) -> str:
        """Re:/Fwd: 접두사를 제거하여 스레드 키 생성"""
        if not subject:
            return "(제목 없음)"
        cleaned = self._THREAD_PREFIX_RE.sub("", subject).strip()
        return cleaned or "(제목 없음)"

    @staticmethod
    def _format_thread(emails: list[Any]) -> str:
        """이메일 스레드를 읽기 쉬운 텍스트로 포맷"""
        parts: list[str] = []
        for email_msg in emails:
            date_str = email_msg.date.strftime("%Y-%m-%d %H:%M") if email_msg.date else ""
            header = f"From: {email_msg.sender}"
            if date_str:
                header += f" ({date_str})"
            header += f"\nTo: {', '.join(email_msg.recipients)}"
            header += f"\nSubject: {email_msg.subject}"

            parts.append(f"{header}\n\n{email_msg.body}")

        return "\n\n---\n\n".join(parts)


# === AttachmentChunker ===


class AttachmentChunker:
    """첨부파일 청커 — DocumentParser로 텍스트 추출 후 DocumentChunker 위임

    PST 첨부파일(Attachment 객체)을 파싱하여 텍스트를 추출하고,
    DocumentChunker로 청킹. 지원하지 않는 파일 형식은 건너뜀.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.doc_chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(
        self,
        attachments: list[Any],
        source_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """첨부파일 리스트를 파싱 후 청킹

        Args:
            attachments: Attachment 객체 리스트
            source_metadata: 첨부파일 출처 메타데이터 (이메일 subject, sender 등)

        Returns:
            Chunk 리스트 (파싱 실패한 첨부파일은 건너뜀)
        """
        from src.parsers.document_parser import DocumentParser

        parser = DocumentParser()
        all_chunks: list[Chunk] = []

        for att in attachments:
            try:
                parsed = parser.parse_bytes(
                    att.content,
                    att.filename,
                    source_metadata=source_metadata,
                )

                # XLSX는 list[ParsedDocument] 반환
                docs = parsed if isinstance(parsed, list) else [parsed]

                for doc in docs:
                    meta = {
                        **doc.metadata,
                        "attachment_filename": att.filename,
                    }
                    if source_metadata:
                        meta["source"] = source_metadata

                    chunks = self.doc_chunker.chunk(doc.content, metadata=meta)
                    # source_type을 attachment로 설정
                    for c in chunks:
                        c.source_type = "attachment"
                    all_chunks.extend(chunks)

            except (ValueError, Exception) as e:
                logger.warning(f"첨부파일 청킹 실패 ({att.filename}): {e}")
                continue

        logger.info(f"첨부파일 청킹 완료: {len(attachments)}개 → {len(all_chunks)}개 청크")
        return all_chunks
