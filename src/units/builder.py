"""파싱 결과 → 증거 단위 (유형별 규칙, 결정적 ID, 정제, 언어 감지, 파트 분할)

유형별 단위 (지시서 §5.1.4)
- 이메일: 메시지 1통 (회신 인용 정제, 전달 메일은 인용 유지)
- 채팅: 같은 방에서 메시지 간격 CHAT_SESSION_GAP_MINUTES(30분) 이하로 이어진 세션
- 문서: PDF 페이지 / PPTX 슬라이드 / DOCX 가상 페이지 / XLSX·XLS 행 블록 / TXT·HTML 가상 페이지
- 첨부: 첨부 1건을 위 형식 규칙으로 페이지·블록 단위로 만든다. 부모(메일·채팅)를 안다.

첨부는 PSTParseResult.attachments(평탄 목록)가 아니라 email.attachments·chat.attachments
(EML/MSG는 ParsedDocument.attachments)를 순회해서 만든다 — 그래야 부모를 안다.

중복 제거(같은 메일이 여러 PST에 있는 경우)는 단위 ID가 같아지는 것을 이용해
`UnitCollector`가 한 곳에서 한다 (멀티프로세싱에서도 동작하도록 빌더와 분리).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any

from src.parsers.document_parser import DocumentParser, ParsedDocument, UnsupportedFormatError
from src.parsers.pst_parser import Attachment, ChatMessage, EmailMessage, PSTParseResult
from src.units import header as hdr
from src.units.email_clean import is_forwarded, strip_quoted
from src.units.lang_detect import detect as detect_lang
from src.units.schema import EvidenceUnit, UnitPart
from src.units.split import docx_blocks, row_blocks, split_text
from src.units.tokens import TokenCounter, get_token_counter
from src.units.topics import extract_topics
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 단위를 만들지 않는 첨부 형식 (건수만 집계)
EXCLUDED_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".emf", ".wmf", ".ico",
    ".hwp", ".hwpx", ".dwg", ".dxf", ".zip", ".7z", ".rar", ".mp4", ".mov", ".avi", ".mp3",
    ".wav", ".exe", ".dll", ".vcf", ".ics",
})
EMAIL_FILE_TYPES = frozenset({"eml", "msg"})
_TAB_RUN_RE = re.compile(r"\t{2,}")


# ─────────────────────────── ID 규칙 ───────────────────────────


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def make_email_id(message_id: str, sender: str = "", date: str = "", subject: str = "",
                  body: str = "") -> str:
    """이메일 ID: em: + sha1(소문자 message_id)[:16], 없으면 sha1(sender|date|subject|본문 앞 200자)"""
    mid = (message_id or "").strip().lower()
    if mid:
        return "em:" + _sha1(mid)[:16]
    return "em:" + _sha1(f"{sender}|{date}|{subject}|{(body or '')[:200]}")[:16]


def make_chat_id(conversation_id: str, start: datetime | None) -> str:
    """채팅 ID: ch: + sha1(conversation_id)[:10] + ':' + 세션 시작 시각(UTC, YYYYMMDDTHHMMSS)"""
    if start is None:
        stamp = "00000000T000000"
    else:
        if start.tzinfo is not None:
            start = start.astimezone(UTC)
        stamp = start.strftime("%Y%m%dT%H%M%S")
    return f"ch:{_sha1(conversation_id)[:10]}:{stamp}"


def make_document_id(rel_path: str, suffix: str) -> str:
    """문서 ID: doc: + sha1(케이스 기준 상대 경로)[:12] + 위치 접미어(:p3, :b2, :s1r2-40)"""
    norm = PurePath(rel_path).as_posix()
    return f"doc:{_sha1(norm)[:12]}{suffix}"


def make_attachment_id(parent_unit_id: str, ordinal: int, suffix: str = "") -> str:
    """첨부 ID: att: + 부모 ID 의 해시 부분 + ':' + 첨부 순번(1부터) + 위치 접미어"""
    parent_hash = parent_unit_id.split(":", 1)[1] if ":" in parent_unit_id else parent_unit_id
    return f"att:{parent_hash}:{ordinal}{suffix}"


# ─────────────────────────── 통계 ───────────────────────────


@dataclass
class BuildStats:
    """단위 생성 통계 (T1 단위 통계의 원천)"""

    counts: Counter = field(default_factory=Counter)
    excluded_types: Counter = field(default_factory=Counter)
    errors: Counter = field(default_factory=Counter)
    # 정제로 줄어든 문자 수 (언어별): {lang: [before, after]}
    clean_chars: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    forwarded_tokens: list[int] = field(default_factory=list)

    def merge(self, other: BuildStats) -> None:
        """다른 통계를 합침 (멀티프로세싱 집계용)"""
        self.counts.update(other.counts)
        self.excluded_types.update(other.excluded_types)
        self.errors.update(other.errors)
        for lang, (b, a) in other.clean_chars.items():
            self.clean_chars[lang][0] += b
            self.clean_chars[lang][1] += a
        self.forwarded_tokens.extend(other.forwarded_tokens)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화용"""
        return {
            "counts": dict(self.counts),
            "excluded_types": dict(self.excluded_types),
            "errors": dict(self.errors),
            "clean_chars": {k: list(v) for k, v in self.clean_chars.items()},
            "forwarded_tokens": list(self.forwarded_tokens),
        }


# ─────────────────────────── 내부 레코드 ───────────────────────────


@dataclass
class _EmailRecord:
    """PST EmailMessage / EML·MSG ParsedDocument 를 같은 모양으로 맞춘 레코드"""

    message_id: str
    sender: str
    sender_name: str
    recipients: list[str]
    recipient_names: list[str]
    cc: list[str]
    cc_names: list[str]
    subject: str
    date: str
    body: str
    attachments: list[Attachment]
    folder_path: str
    in_reply_to: str = ""


def _iso(dt: datetime | None) -> str:
    return dt.isoformat() if dt else ""


def _email_body_from_parsed(parsed: ParsedDocument) -> str:
    """EML/MSG 파서가 본문 앞에 붙인 'From: …' 헤더 블록을 떼어낸 본문"""
    content = parsed.content or ""
    first_line = content.lstrip().split("\n", 1)[0]
    if not first_line.lower().startswith("from:"):
        return content
    parts = content.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else ""


# ─────────────────────────── 빌더 ───────────────────────────


class UnitBuilder:
    """파싱 결과를 증거 단위로 변환

    Args:
        case_id: 케이스 ID (모든 단위 meta 에 기록)
        counter: 토큰 카운터 (기본: settings 기준 싱글턴)
        max_part_tokens: 파트 상한 (기본: settings.unit_max_part_tokens)
        clean_enabled: 이메일 정제 여부 (기본: settings.email_clean_enabled)
        gap_minutes: 채팅 세션 간격 (기본: settings.chat_session_gap_minutes)
        doc_parser: 첨부 파싱에 쓸 DocumentParser
    """

    def __init__(
        self,
        case_id: str,
        *,
        counter: TokenCounter | None = None,
        max_part_tokens: int | None = None,
        clean_enabled: bool | None = None,
        gap_minutes: int | None = None,
        doc_parser: DocumentParser | None = None,
    ) -> None:
        from src.utils.config import settings

        self.case_id = case_id
        self.counter = counter or get_token_counter()
        self.max_part_tokens = max_part_tokens or settings.unit_max_part_tokens
        self.clean_enabled = (
            settings.email_clean_enabled if clean_enabled is None else clean_enabled
        )
        self.gap_minutes = gap_minutes or settings.chat_session_gap_minutes
        self.doc_parser = doc_parser or DocumentParser()
        self.stats = BuildStats()

    # ── 공통: 메타 마무리 + 파트 분할 ──

    def _finish(self, unit_id: str, unit_type: str, text: str, meta: dict[str, Any]) -> EvidenceUnit:
        """언어 감지·토픽·헤더·파트를 채워 EvidenceUnit 생성"""
        text = (text or "").strip()
        lang = detect_lang(text)
        meta = {
            "case_id": self.case_id,
            **meta,
            "lang": lang.lang,
            "lang_mixed": lang.mixed,
            "lang_second": lang.second,
            "topics": extract_topics(text),
        }
        header = hdr.build_header(unit_type, meta)
        pieces = split_text(text, self.max_part_tokens, self.counter)
        # 본문이 빈 단위(첨부만 있는 메일 등)도 헤더로 검색되게 파트 1개를 둔다
        pieces_data = [(p.text, p.n_tokens) for p in pieces] if pieces else [("", 0)]
        parts = [
            UnitPart(
                part_id=f"{unit_id}#{k}",
                unit_id=unit_id,
                unit_type=unit_type,
                header=header,
                text=t,
                n_tokens=n,
                meta=meta,
            )
            for k, (t, n) in enumerate(pieces_data, 1)
        ]
        self.stats.counts[f"units:{unit_type}"] += 1
        self.stats.counts[f"parts:{unit_type}"] += len(parts)
        if len(parts) > 1:
            self.stats.counts[f"split_units:{unit_type}"] += 1
        return EvidenceUnit(unit_id, unit_type, header, text, meta, parts)

    # ── 진입점: 파일 ──

    def build_file(self, path: str | Path, rel_path: str) -> list[EvidenceUnit]:
        """파일 하나 → 단위 목록 (EML/MSG는 이메일, PST/OST는 메일함, 그 밖은 문서)

        Args:
            path: 파일 경로
            rel_path: 케이스 기준 상대 경로 (문서 ID와 source_file 에 쓰임)
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".pst", ".ost"):
            from src.parsers.pst_parser import PSTParser

            return self.build_from_pst(PSTParser(path).parse(), rel_path)
        parsed = self.doc_parser.parse(path)
        docs = parsed if isinstance(parsed, list) else [parsed]
        if docs and docs[0].file_type in EMAIL_FILE_TYPES:
            folder = PurePath(rel_path).parent.as_posix()
            return self.build_from_parsed_email(docs[0], rel_path, folder_path=folder)
        return self.build_document(docs, rel_path)

    # ── 이메일 ──

    def build_from_parsed_email(
        self, parsed: ParsedDocument, source_file: str, folder_path: str = ""
    ) -> list[EvidenceUnit]:
        """EML/MSG ParsedDocument → 이메일 단위 + 첨부 단위"""
        m = parsed.metadata
        rec = _EmailRecord(
            message_id=m.get("message_id", ""),
            sender=m.get("sender", ""),
            sender_name=m.get("sender_name", ""),
            recipients=list(m.get("recipients") or []),
            recipient_names=list(m.get("recipient_names") or []),
            cc=list(m.get("cc") or []),
            cc_names=list(m.get("cc_names") or []),
            subject=m.get("subject", ""),
            date=m.get("date", ""),
            body=_email_body_from_parsed(parsed),
            attachments=list(parsed.attachments),
            folder_path=folder_path,
            in_reply_to=m.get("in_reply_to", ""),
        )
        return self._build_email(rec, source_file)

    def _build_email(self, rec: _EmailRecord, source_file: str) -> list[EvidenceUnit]:
        unit_id = make_email_id(rec.message_id, rec.sender, rec.date, rec.subject, rec.body)
        forwarded = is_forwarded(rec.subject)
        text = strip_quoted(rec.body, rec.subject) if self.clean_enabled else rec.body
        meta: dict[str, Any] = {
            "source_file": PurePath(source_file).as_posix(),
            "date": rec.date,
            "date_end": "",
            "participants": _unique([rec.sender, *rec.recipients, *rec.cc]),
            "sender": rec.sender,
            "sender_name": rec.sender_name,
            "recipients": rec.recipients,
            "recipient_names": rec.recipient_names,
            "cc": rec.cc,
            "cc_names": rec.cc_names,
            "subject": rec.subject,
            "message_id": rec.message_id,
            "in_reply_to": rec.in_reply_to,
            "attachment_filenames": [a.filename for a in rec.attachments],
            "folder_paths": [rec.folder_path] if rec.folder_path else [],
            "forwarded": forwarded,
        }
        unit = self._finish(unit_id, "email", text, meta)
        # 정제 통계 (언어별 문자 수 전후)
        before, after = len((rec.body or "").strip()), len(unit.text)
        self.stats.clean_chars[unit.meta["lang"]][0] += before
        self.stats.clean_chars[unit.meta["lang"]][1] += after
        if forwarded:
            self.stats.counts["forwarded_emails"] += 1
            self.stats.forwarded_tokens.append(unit.n_tokens)

        parent = {
            "parent_unit_id": unit_id,
            "parent_kind": "email",
            "parent_subject": rec.subject,
            "parent_sender": rec.sender_name or rec.sender,
            "parent_recipients": rec.recipient_names or rec.recipients,
            "parent_date": rec.date,
            "date": rec.date,
            "participants": meta["participants"],
        }
        units = [unit]
        for ordinal, att in enumerate(rec.attachments, 1):
            units.extend(self._build_attachment(att, ordinal, parent, source_file))
        return units

    # ── PST (이메일 + 채팅 + 첨부) ──

    def build_from_pst(self, result: PSTParseResult, source_file: str) -> list[EvidenceUnit]:
        """PSTParseResult → 이메일·채팅 세션·첨부 단위

        첨부는 email.attachments·chat.attachments 를 순회해 부모를 붙인다. 평탄 목록
        (result.attachments)에만 있고 부모를 찾을 수 없는 첨부는 parent_unit_id="" 로 만든다.
        """
        units: list[EvidenceUnit] = []
        linked: set[int] = set()
        for em in result.emails:
            linked.update(id(a) for a in em.attachments)
            units.extend(self._build_email(self._record_from_pst(em), source_file))

        units.extend(self._build_chats(result.chats, source_file, linked))

        orphans = [a for a in result.attachments if id(a) not in linked]
        for ordinal, att in enumerate(orphans, 1):
            self.stats.counts["orphan_attachments"] += 1
            orphan_parent = {
                "parent_unit_id": "",
                "orphan_key": make_document_id(source_file, f":orphan{ordinal}"),
            }
            units.extend(self._build_attachment(att, ordinal, orphan_parent, source_file))
        return units

    @staticmethod
    def _record_from_pst(em: EmailMessage) -> _EmailRecord:
        return _EmailRecord(
            message_id=em.message_id,
            sender=em.sender,
            sender_name=em.sender,
            recipients=list(em.recipients),
            recipient_names=list(em.recipients),
            cc=list(em.cc),
            cc_names=list(em.cc),
            subject=em.subject,
            date=_iso(em.date),
            body=em.body,
            attachments=list(em.attachments),
            folder_path=em.folder_path,
            in_reply_to=em.in_reply_to,
        )

    def _build_chats(
        self, chats: list[ChatMessage], source_file: str, linked: set[int]
    ) -> list[EvidenceUnit]:
        """채팅 메시지 → 방(conversation_id)별 30분 세션 단위 + 첨부"""
        by_room: dict[str, list[ChatMessage]] = defaultdict(list)
        for c in chats:
            by_room[c.conversation_id or c.folder_path or "unknown"].append(c)

        units: list[EvidenceUnit] = []
        for room_key, msgs in by_room.items():
            for session in self._split_sessions(msgs):
                units.extend(self._build_chat_session(room_key, session, source_file, linked))
        return units

    def _split_sessions(self, msgs: list[ChatMessage]) -> list[list[ChatMessage]]:
        """메시지 간격이 gap_minutes 를 넘으면 새 세션 (타임스탬프 없는 메시지는 직전 세션에 붙임)"""
        ordered = sorted(msgs, key=lambda m: _sort_key(m.timestamp))
        sessions: list[list[ChatMessage]] = []
        for msg in ordered:
            if sessions:
                prev = sessions[-1][-1].timestamp
                cur = msg.timestamp
                if prev and cur and _minutes_between(prev, cur) <= self.gap_minutes:
                    sessions[-1].append(msg)
                    continue
                if not cur or not prev:
                    sessions[-1].append(msg)
                    continue
            sessions.append([msg])
        return sessions

    def _build_chat_session(
        self, room_key: str, session: list[ChatMessage], source_file: str, linked: set[int]
    ) -> list[EvidenceUnit]:
        stamps = [m.timestamp for m in session if m.timestamp]
        start = min(stamps, key=_sort_key) if stamps else None
        end = max(stamps, key=_sort_key) if stamps else None
        unit_id = make_chat_id(room_key, start)
        participants = _unique([p for m in session for p in [m.sender, *m.participants]])
        lines = [
            f"{m.timestamp:%H:%M} {m.sender}: {m.body}".strip() if m.timestamp
            else f"{m.sender}: {m.body}".strip()
            for m in session
        ]
        first = session[0]
        room = first.conversation_topic or PurePath(first.folder_path or room_key).name
        meta: dict[str, Any] = {
            "source_file": PurePath(source_file).as_posix(),
            "date": _iso(start),
            "date_end": _iso(end),
            "participants": participants,
            "conversation_id": room_key,
            "room": room,
            "chat_type": "group" if len(participants) > 2 else "1:1",
            "n_messages": len(session),
        }
        units = [self._finish(unit_id, "chat", "\n".join(lines), meta)]
        ordinal = 0
        for m in session:
            for att in m.attachments:
                ordinal += 1
                linked.add(id(att))
                parent = {
                    "parent_unit_id": unit_id,
                    "parent_kind": "chat",
                    "parent_subject": room,
                    "parent_sender": m.sender,
                    "parent_recipients": [],
                    "parent_date": _iso(m.timestamp),
                    "date": _iso(m.timestamp),
                    "participants": participants,
                }
                units.extend(self._build_attachment(att, ordinal, parent, source_file))
        return units

    # ── 첨부 ──

    def _build_attachment(
        self, att: Attachment, ordinal: int, parent: dict[str, Any], source_file: str
    ) -> list[EvidenceUnit]:
        """첨부 1건 → 형식 규칙에 따른 페이지·블록 단위 (지원하지 않는 형식은 건수만 집계)"""
        ext = Path(att.filename or "").suffix.lower()
        self.stats.counts["attachments_total"] += 1
        if ext in EXCLUDED_EXTENSIONS:
            self.stats.excluded_types[ext] += 1
            return []
        if not att.content:
            self.stats.excluded_types["(empty)"] += 1
            return []
        try:
            parsed = self.doc_parser.parse_bytes(att.content, att.filename or "unnamed")
        except UnsupportedFormatError:
            self.stats.excluded_types[ext or "(none)"] += 1
            return []
        except Exception as e:
            self.stats.errors[f"attachment_parse:{ext or '(none)'}"] += 1
            logger.warning(f"첨부 파싱 실패 ({att.filename}): {e}")
            return []

        docs = parsed if isinstance(parsed, list) else [parsed]
        parent_id = parent.get("parent_unit_id") or ""
        base_key = parent_id or parent.get("orphan_key", "")

        def id_for(suffix: str) -> str:
            return make_attachment_id(base_key, ordinal, suffix)

        base_meta = {
            "source_file": PurePath(source_file).as_posix(),
            "date": parent.get("date", ""),
            "date_end": "",
            "participants": parent.get("participants", []),
            "filename": att.filename,
            **{k: v for k, v in parent.items() if k.startswith("parent_")},
            "parent_unit_id": parent_id,
        }
        self.stats.counts["attachments_parsed"] += 1
        return self._document_units(docs, base_meta, "attachment", id_for)

    # ── 문서 ──

    def build_document(self, docs: list[ParsedDocument], rel_path: str) -> list[EvidenceUnit]:
        """문서 파일(시트별 리스트 포함) → 페이지·블록 단위"""
        if not docs:
            return []
        first = docs[0].metadata
        base_meta = {
            "source_file": PurePath(rel_path).as_posix(),
            "date": first.get("last_modified") or first.get("modified_date")
            or first.get("created_date") or "",
            "date_end": "",
            "participants": _unique([first.get("author", ""), first.get("last_modified_by", "")]),
            "filename": first.get("filename") or PurePath(rel_path).name,
        }
        return self._document_units(
            docs, base_meta, "document", lambda suffix: make_document_id(rel_path, suffix)
        )

    def _document_units(
        self,
        docs: list[ParsedDocument],
        base_meta: dict[str, Any],
        unit_type: str,
        id_for: Any,
    ) -> list[EvidenceUnit]:
        """형식별 페이지·블록 규칙 (문서와 첨부 공용)"""
        doc0 = docs[0]
        ftype = doc0.file_type
        common = {
            **base_meta,
            "file_type": ftype,
            "author": doc0.metadata.get("author", ""),
            "last_modified_by": doc0.metadata.get("last_modified_by", ""),
            "created_date": doc0.metadata.get("created_date", ""),
            "last_modified": doc0.metadata.get("last_modified")
            or doc0.metadata.get("modified_date", ""),
        }
        units: list[EvidenceUnit] = []

        if ftype in ("xlsx", "xls"):
            for sheet_idx, sheet in enumerate(docs, 1):
                name = sheet.metadata.get("sheet_name", str(sheet_idx))
                # 빈 셀이 많은 시트의 연속 탭은 토큰만 늘리므로 하나로 줄인다
                rows = [_TAB_RUN_RE.sub("\t", r) for r in sheet.content.split("\n")] \
                    if sheet.content else []
                for blk in row_blocks(rows, sheet.row_numbers, self.max_part_tokens, self.counter):
                    meta = {
                        **common,
                        "page": 0,
                        "page_count": 0,
                        "page_kind": "sheet",
                        "sheet_name": name,
                        "row_start": blk.start_row,
                        "row_end": blk.end_row,
                        "position_extra": f"시트 {name} {blk.start_row}~{blk.end_row}행",
                    }
                    uid = id_for(f":s{sheet_idx}r{blk.start_row}-{blk.end_row}")
                    units.append(self._finish(uid, unit_type, blk.text, meta))
            return units

        if ftype in ("pdf", "pptx"):
            kind = "page" if ftype == "pdf" else "slide"
            pages = doc0.pages
            if doc0.metadata.get("ocr_pages"):
                self.stats.counts["ocr_pages"] += int(doc0.metadata["ocr_pages"])
            for i, page_text in enumerate(pages, 1):
                if not page_text.strip():
                    self.stats.counts[f"empty_pages:{ftype}"] += 1
                    continue
                meta = {**common, "page": i, "page_count": len(pages), "page_kind": kind}
                units.append(self._finish(id_for(f":p{i}"), unit_type, page_text, meta))
            return units

        if ftype in EMAIL_FILE_TYPES:
            # 첨부된 메일: 헤더 줄을 포함한 본문 전체를 가상 페이지로 (중첩 첨부는 펼치지 않음)
            self.stats.counts["nested_email_attachments"] += 1
            blocks = split_text(doc0.content, self.max_part_tokens, self.counter)
        elif ftype == "docx":
            blocks = docx_blocks(doc0.pages or [doc0.content], self.max_part_tokens, self.counter)
        else:  # txt, html
            blocks = split_text(doc0.content, self.max_part_tokens, self.counter)

        for i, blk in enumerate(blocks, 1):
            meta = {**common, "page": i, "page_count": len(blocks), "page_kind": "block"}
            units.append(self._finish(id_for(f":b{i}"), unit_type, blk.text, meta))
        return units


# ─────────────────────────── 중복 제거 ───────────────────────────


class UnitCollector:
    """단위 수집 + unit_id 기준 중복 제거

    같은 메일이 조사 대상자 여러 명의 PST에 있으면 같은 unit_id가 된다. 처음 것만 남기고
    이메일은 folder_paths 를 병합한다. 첨부 ID는 부모 ID에서 나오므로 함께 중복 제거된다.
    """

    def __init__(self) -> None:
        self.units: dict[str, EvidenceUnit] = {}
        self.duplicates: Counter = Counter()

    def add(self, units: list[EvidenceUnit]) -> None:
        """단위 목록 추가 (중복은 병합 후 버림)"""
        for u in units:
            kept = self.units.get(u.unit_id)
            if kept is None:
                self.units[u.unit_id] = u
                continue
            self.duplicates[u.unit_type] += 1
            if u.unit_type == "email":
                merged = _unique([*kept.meta.get("folder_paths", []),
                                  *u.meta.get("folder_paths", [])])
                kept.meta["folder_paths"] = merged
                for p in kept.parts:
                    p.meta["folder_paths"] = merged

    def parts(self) -> list[UnitPart]:
        """수집된 모든 파트 (단위 추가 순서)"""
        return [p for u in self.units.values() for p in u.parts]


# ─────────────────────────── 헬퍼 ───────────────────────────


def _unique(items: list[str]) -> list[str]:
    """순서 유지 중복 제거 (대소문자 무시, 빈 값 제외)"""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = (item or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _sort_key(dt: datetime | None) -> float:
    """naive/aware 혼재 정렬용 (naive 는 UTC로 간주)"""
    if dt is None:
        return float("-inf")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _minutes_between(a: datetime, b: datetime) -> float:
    return abs(_sort_key(b) - _sort_key(a)) / 60.0
