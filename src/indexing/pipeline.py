"""인덱싱 파이프라인 오케스트레이터

케이스 단위로 PST/문서 파일을 파싱 → 청킹 → 메타데이터 → 벡터 저장하는
전체 Phase A 파이프라인을 관리.

파이프라인 단계:
    1. 파일 수집 (PST + 문서 경로 스캔)
    2. 파싱 (PST → 이메일/채팅/첨부, 문서 → 텍스트)
    3. 청킹 (소스 타입별 최적 분할)
    4. 메타데이터 보강 (case_id, topics)
    5. 벡터 저장 (ChromaDB + BM25)

진행률은 IndexingProgress 객체로 실시간 추적 가능.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.cases.case_store import CaseMetadata, CaseStatus, CaseStore
from src.chunkers.chunker import (
    AttachmentChunker,
    ChatChunker,
    Chunk,
    DocumentChunker,
    EmailChunker,
)
from src.chunkers.metadata_enricher import enrich_chunks
from src.parsers.document_parser import DocumentParser
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 문서 파서가 지원하는 확장자
_DOCUMENT_EXTENSIONS = DocumentParser.SUPPORTED_TYPES


class IndexingPhase(str, Enum):
    """인덱싱 진행 단계"""

    IDLE = "idle"
    SCANNING = "scanning"
    PARSING = "parsing"
    CHUNKING = "chunking"
    ENRICHING = "enriching"
    STORING = "storing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class IndexingProgress:
    """인덱싱 진행 상황 (실시간 추적)"""

    case_id: str
    phase: IndexingPhase = IndexingPhase.IDLE
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def progress_percent(self) -> float:
        """진행률 (0.0 ~ 100.0)"""
        if self.total_files == 0:
            return 0.0
        return min(100.0, (self.processed_files / self.total_files) * 100)

    @property
    def is_running(self) -> bool:
        return self.phase not in (
            IndexingPhase.IDLE,
            IndexingPhase.COMPLETED,
            IndexingPhase.ERROR,
            IndexingPhase.CANCELLED,
        )

    def to_dict(self) -> dict[str, Any]:
        """API 응답용 딕셔너리"""
        elapsed = ""
        if self.started_at:
            end = self.completed_at or datetime.now()
            secs = int((end - self.started_at).total_seconds())
            mins, secs = divmod(secs, 60)
            elapsed = f"{mins}분 {secs}초" if mins else f"{secs}초"

        return {
            "case_id": self.case_id,
            "status": self.phase.value,
            "phase": self.phase.value,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "total_chunks": self.total_chunks,
            "progress_percent": round(self.progress_percent, 1),
            "elapsed": elapsed,
            "errors": self.errors,
        }


class IndexingPipeline:
    """케이스 기반 인덱싱 파이프라인 오케스트레이터

    사용법:
        pipeline = IndexingPipeline(case_store)
        progress = pipeline.run(case_id)
        # progress.phase == IndexingPhase.COMPLETED

        # 비동기 실행
        pipeline.run_async(case_id)
        progress = pipeline.get_progress(case_id)
    """

    def __init__(
        self,
        case_store: CaseStore | None = None,
        vector_store_factory: Any = None,
    ) -> None:
        """IndexingPipeline 초기화

        Args:
            case_store: 케이스 저장소 (기본: 새 CaseStore 생성)
            vector_store_factory: VectorStoreService 생성 팩토리 (테스트용 주입)
        """
        self.case_store = case_store or CaseStore()
        self._vector_store_factory = vector_store_factory
        self._doc_parser = DocumentParser()
        self._doc_chunker = DocumentChunker()
        self._chat_chunker = ChatChunker()
        self._email_chunker = EmailChunker()
        self._attachment_chunker = AttachmentChunker()

        # 진행률 추적 (case_id → IndexingProgress)
        self._progress: dict[str, IndexingProgress] = {}
        # 실행 중인 스레드 (case_id → Thread)
        self._threads: dict[str, threading.Thread] = {}
        # 취소 플래그 (case_id → Event)
        self._cancel_flags: dict[str, threading.Event] = {}

    def _create_vector_store(self, case_id: str) -> Any:
        """케이스별 VectorStoreService 생성"""
        if self._vector_store_factory:
            return self._vector_store_factory(case_id)

        from src.vectorstore.vector_store import VectorStoreService

        return VectorStoreService(collection_name=f"case_{case_id}")

    def get_progress(self, case_id: str) -> IndexingProgress:
        """케이스 인덱싱 진행률 조회"""
        if case_id not in self._progress:
            return IndexingProgress(case_id=case_id)
        return self._progress[case_id]

    def is_running(self, case_id: str) -> bool:
        """케이스 인덱싱 실행 중 여부"""
        return case_id in self._progress and self._progress[case_id].is_running

    def cancel(self, case_id: str) -> bool:
        """인덱싱 중단 요청

        Returns:
            True: 중단 요청 성공, False: 실행 중이 아님
        """
        if not self.is_running(case_id):
            return False
        if case_id in self._cancel_flags:
            self._cancel_flags[case_id].set()
            logger.info(f"인덱싱 중단 요청: {case_id}")
            return True
        return False

    def run_async(self, case_id: str) -> IndexingProgress:
        """백그라운드 스레드로 인덱싱 실행

        Args:
            case_id: 인덱싱할 케이스 ID

        Returns:
            IndexingProgress (실행 시작 상태)
        """
        if self.is_running(case_id):
            return self._progress[case_id]

        cancel_flag = threading.Event()
        self._cancel_flags[case_id] = cancel_flag

        thread = threading.Thread(
            target=self._run_safe,
            args=(case_id, cancel_flag),
            daemon=True,
            name=f"indexing-{case_id}",
        )
        self._threads[case_id] = thread
        thread.start()

        # 스레드 시작 후 progress가 생성될 때까지 잠시 대기
        for _ in range(10):
            if case_id in self._progress:
                break
            time.sleep(0.05)

        return self.get_progress(case_id)

    def _run_safe(self, case_id: str, cancel_flag: threading.Event) -> None:
        """예외 안전 래퍼"""
        try:
            self.run(case_id, cancel_flag=cancel_flag)
        except Exception as e:
            logger.error(f"인덱싱 파이프라인 실패: {case_id} — {e}")
            if case_id in self._progress:
                self._progress[case_id].phase = IndexingPhase.ERROR
                self._progress[case_id].errors.append(str(e))

    def run(
        self,
        case_id: str,
        cancel_flag: threading.Event | None = None,
    ) -> IndexingProgress:
        """동기 인덱싱 실행

        전체 파이프라인: 파일 수집 → 파싱 → 청킹 → 메타데이터 → 벡터 저장

        Args:
            case_id: 인덱싱할 케이스 ID
            cancel_flag: 취소 플래그 (set되면 중단)

        Returns:
            최종 IndexingProgress
        """
        progress = IndexingProgress(
            case_id=case_id,
            started_at=datetime.now(),
        )
        self._progress[case_id] = progress

        try:
            # 케이스 조회 + 상태 전이
            case_meta = self.case_store.get(case_id)
            self.case_store.update_status(case_id, CaseStatus.INDEXING)

            # 1. 파일 수집
            progress.phase = IndexingPhase.SCANNING
            files = self._collect_files(case_meta)
            progress.total_files = len(files)
            logger.info(f"파일 수집 완료: {len(files)}개 ({case_id})")

            if not files:
                progress.phase = IndexingPhase.COMPLETED
                progress.completed_at = datetime.now()
                self.case_store.update_status(case_id, CaseStatus.READY)
                self.case_store.update_stats(case_id, total_documents=0, total_chunks=0)
                return progress

            # 2~4. 파싱 → 청킹 → 메타데이터 보강
            all_chunks: list[Chunk] = []

            for file_path in files:
                if cancel_flag and cancel_flag.is_set():
                    progress.phase = IndexingPhase.CANCELLED
                    self.case_store.update_status(case_id, CaseStatus.CREATED)
                    logger.info(f"인덱싱 취소됨: {case_id}")
                    return progress

                try:
                    chunks = self._process_file(file_path, case_id, progress)
                    all_chunks.extend(chunks)
                except Exception as e:
                    error_msg = f"파일 처리 실패 ({file_path.name}): {e}"
                    progress.errors.append(error_msg)
                    logger.warning(error_msg)
                finally:
                    progress.processed_files += 1

            # 5. 벡터 저장
            progress.phase = IndexingPhase.STORING
            stored_count = 0
            if all_chunks:
                vector_store = self._create_vector_store(case_id)
                # 배치 단위로 저장
                batch_size = 100
                for i in range(0, len(all_chunks), batch_size):
                    if cancel_flag and cancel_flag.is_set():
                        progress.phase = IndexingPhase.CANCELLED
                        self.case_store.update_status(case_id, CaseStatus.CREATED)
                        return progress

                    batch = all_chunks[i : i + batch_size]
                    try:
                        stored_count += vector_store.add_chunks(batch)
                    except Exception as e:
                        error_msg = f"벡터 저장 실패 (batch {i // batch_size}): {e}"
                        progress.errors.append(error_msg)
                        logger.warning(error_msg)

            progress.total_chunks = stored_count
            progress.phase = IndexingPhase.COMPLETED
            progress.completed_at = datetime.now()

            # 케이스 통계 업데이트 + 상태 변경
            self.case_store.update_stats(
                case_id,
                total_documents=progress.processed_files,
                total_chunks=stored_count,
            )
            self.case_store.update_status(case_id, CaseStatus.READY)

            logger.info(
                f"인덱싱 완료: {case_id} — "
                f"파일 {progress.processed_files}개, 청크 {stored_count}개, "
                f"에러 {len(progress.errors)}개"
            )
            return progress

        except Exception as e:
            progress.phase = IndexingPhase.ERROR
            progress.errors.append(str(e))
            progress.completed_at = datetime.now()
            try:
                self.case_store.set_error(case_id, str(e))
            except Exception:
                pass  # 케이스 자체가 없을 수 있음
            logger.error(f"인덱싱 파이프라인 에러: {case_id} — {e}")
            return progress

    def _collect_files(self, case_meta: CaseMetadata) -> list[Path]:
        """케이스 데이터 소스에서 처리 대상 파일 수집

        PST 파일은 직접 추가, 문서 경로는 재귀 스캔.
        """
        files: list[Path] = []

        # PST 파일
        for pst_path_str in case_meta.pst_paths:
            pst_path = Path(pst_path_str)
            if pst_path.is_file() and pst_path.suffix.lower() in (".pst", ".ost"):
                files.append(pst_path)
            elif not pst_path.exists():
                logger.warning(f"PST 파일 없음: {pst_path}")

        # 문서 폴더 재귀 스캔
        for doc_path_str in case_meta.doc_paths:
            doc_path = Path(doc_path_str)
            if doc_path.is_file():
                if doc_path.suffix.lower() in _DOCUMENT_EXTENSIONS:
                    files.append(doc_path)
            elif doc_path.is_dir():
                for ext in _DOCUMENT_EXTENSIONS:
                    files.extend(doc_path.rglob(f"*{ext}"))
            else:
                logger.warning(f"문서 경로 없음: {doc_path}")

        # 중복 제거 (resolve로 정규화)
        seen: set[Path] = set()
        unique_files: list[Path] = []
        for f in files:
            resolved = f.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_files.append(f)

        return unique_files

    def _process_file(
        self,
        file_path: Path,
        case_id: str,
        progress: IndexingProgress,
    ) -> list[Chunk]:
        """단일 파일 처리: 파싱 → 청킹 → 메타데이터 보강

        PST 파일은 이메일/채팅/첨부를 각각 적절한 청커로 처리.
        일반 문서는 DocumentChunker로 처리.
        """
        suffix = file_path.suffix.lower()

        if suffix in (".pst", ".ost"):
            return self._process_pst(file_path, case_id, progress)
        else:
            return self._process_document(file_path, case_id, progress)

    def _process_pst(
        self,
        pst_path: Path,
        case_id: str,
        progress: IndexingProgress,
    ) -> list[Chunk]:
        """PST 파일 처리: 파싱 → 이메일/채팅/첨부 청킹"""
        from src.parsers.pst_parser import PSTParser

        progress.phase = IndexingPhase.PARSING
        parser = PSTParser(pst_path)
        result = parser.parse()

        all_chunks: list[Chunk] = []
        pst_meta = {"pst_file": pst_path.name}

        # 이메일 청킹
        progress.phase = IndexingPhase.CHUNKING
        if result.emails:
            email_chunks = self._email_chunker.chunk(result.emails, metadata=pst_meta)
            all_chunks.extend(email_chunks)

        # 채팅 청킹
        if result.chats:
            chat_chunks = self._chat_chunker.chunk_chat_messages(
                result.chats, extra_metadata=pst_meta
            )
            all_chunks.extend(chat_chunks)

        # 첨부파일 청킹
        if result.attachments:
            att_chunks = self._attachment_chunker.chunk(
                result.attachments, source_metadata=pst_meta
            )
            all_chunks.extend(att_chunks)

        # 메타데이터 보강
        progress.phase = IndexingPhase.ENRICHING
        if all_chunks:
            enrich_chunks(all_chunks, case_id=case_id)

        logger.info(
            f"PST 처리 완료: {pst_path.name} → "
            f"이메일 {len(result.emails)}, 채팅 {len(result.chats)}, "
            f"첨부 {len(result.attachments)}, 청크 {len(all_chunks)}"
        )
        return all_chunks

    def _process_document(
        self,
        doc_path: Path,
        case_id: str,
        progress: IndexingProgress,
    ) -> list[Chunk]:
        """문서 파일 처리: 파싱 → 청킹 → 메타데이터 보강"""
        progress.phase = IndexingPhase.PARSING
        parsed = self._doc_parser.parse(doc_path)

        progress.phase = IndexingPhase.CHUNKING
        # XLSX는 list[ParsedDocument] 반환
        docs = parsed if isinstance(parsed, list) else [parsed]

        all_chunks: list[Chunk] = []
        for doc in docs:
            chunks = self._doc_chunker.chunk_parsed_document(doc)
            all_chunks.extend(chunks)

        # 메타데이터 보강
        progress.phase = IndexingPhase.ENRICHING
        if all_chunks:
            enrich_chunks(all_chunks, case_id=case_id)

        logger.info(f"문서 처리 완료: {doc_path.name} → {len(all_chunks)}개 청크")
        return all_chunks
