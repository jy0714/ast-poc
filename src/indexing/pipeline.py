"""인덱싱 파이프라인 오케스트레이터

케이스 단위로 PST/문서 파일을 파싱 → 청킹 → 메타데이터 → 벡터 저장하는
전체 Phase A 파이프라인을 관리.

파이프라인 단계:
    1. 파일 수집 (PST + 문서 경로 스캔)
    2. 파싱 (PST → 이메일/채팅/첨부, 문서 → 텍스트)  [멀티프로세싱]
    3. 청킹 (소스 타입별 최적 분할)                     [멀티프로세싱]
    4. 메타데이터 보강 (case_id, topics)                [멀티프로세싱]
    5. 벡터 저장 (ChromaDB + BM25)                     [GPU 배치]

대용량(수백만 파일) 처리 시 CPU 코어를 모두 활용하여
파싱+청킹을 병렬 처리하고, GPU는 임베딩에 집중합니다.

진행률은 IndexingProgress 객체로 실시간 추적 가능.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _get_worker_count() -> int:
    """파싱/청킹 워커 수 결정"""
    if settings.indexing_workers > 0:
        return settings.indexing_workers
    return max(1, os.cpu_count() or 4)


def _process_file_worker(args: tuple[str, str]) -> list[dict[str, Any]]:
    """멀티프로세싱 워커: 단일 파일 → 파싱 → 청킹 → 메타데이터 보강 → 직렬화된 청크 반환

    별도 프로세스에서 실행되므로 Chunk 객체를 dict로 직렬화하여 반환.
    """
    file_path_str, case_id = args
    file_path = Path(file_path_str)

    try:
        suffix = file_path.suffix.lower()
        doc_parser = DocumentParser()

        if suffix in (".pst", ".ost"):
            chunks = _process_pst_standalone(file_path, case_id, doc_parser)
        else:
            chunks = _process_document_standalone(file_path, case_id, doc_parser)

        if chunks:
            enrich_chunks(chunks, case_id=case_id)

        return [
            {
                "content": c.content,
                "metadata": c.metadata,
                "chunk_id": c.chunk_id,
                "source_type": c.source_type,
            }
            for c in chunks
        ]
    except Exception as e:
        return [{"__error__": f"파일 처리 실패 ({file_path.name}): {e}"}]


def _process_pst_standalone(
    pst_path: Path, case_id: str, doc_parser: DocumentParser
) -> list[Chunk]:
    """독립 프로세스용 PST 처리"""
    from src.parsers.pst_parser import PSTParser

    parser = PSTParser(pst_path)
    result = parser.parse()

    all_chunks: list[Chunk] = []
    pst_meta = {"pst_file": pst_path.name}

    if result.emails:
        email_chunker = EmailChunker()
        all_chunks.extend(email_chunker.chunk(result.emails, metadata=pst_meta))

    if result.chats:
        chat_chunker = ChatChunker()
        all_chunks.extend(chat_chunker.chunk_chat_messages(result.chats, extra_metadata=pst_meta))

    if result.attachments:
        att_chunker = AttachmentChunker()
        all_chunks.extend(att_chunker.chunk(result.attachments, source_metadata=pst_meta))

    return all_chunks


def _process_document_standalone(
    doc_path: Path, case_id: str, doc_parser: DocumentParser
) -> list[Chunk]:
    """독립 프로세스용 문서 처리"""
    parsed = doc_parser.parse(doc_path)
    docs = parsed if isinstance(parsed, list) else [parsed]

    doc_chunker = DocumentChunker()
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(doc_chunker.chunk_parsed_document(doc))

    return all_chunks

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
        """동기 인덱싱 실행 (CPU/GPU 동시 스트리밍 파이프라인)

        CPU와 GPU를 동시에 활용하는 Producer-Consumer 구조:
            [CPU 워커들: 파싱+청킹] ──큐──→ [GPU 스레드: 임베딩+저장]

        파일 수가 적을 때는 오버헤드 없이 순차 처리로 fallback.

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

            # 2~5. 스트리밍 파이프라인 (CPU 파싱 + GPU 임베딩 동시 실행)
            num_workers = _get_worker_count()
            use_streaming = len(files) > num_workers * 2 and num_workers > 1

            if use_streaming:
                progress.phase = IndexingPhase.PARSING
                stored_count = self._run_streaming_pipeline(
                    files, case_id, progress, cancel_flag,
                )
            else:
                # 파일 수가 적으면 순차 처리
                progress.phase = IndexingPhase.PARSING
                all_chunks = self._sequential_process_files(
                    files, case_id, progress, cancel_flag,
                )
                if cancel_flag and cancel_flag.is_set():
                    progress.phase = IndexingPhase.CANCELLED
                    progress.completed_at = datetime.now()
                    self.case_store.update_status(case_id, CaseStatus.CREATED)
                    _save_indexing_log(progress, "cancelled")
                    return progress

                progress.phase = IndexingPhase.STORING
                stored_count = self._store_chunks_batch(
                    all_chunks, case_id, cancel_flag,
                )

            if cancel_flag and cancel_flag.is_set():
                progress.phase = IndexingPhase.CANCELLED
                progress.completed_at = datetime.now()
                self.case_store.update_status(case_id, CaseStatus.CREATED)
                _save_indexing_log(progress, "cancelled")
                logger.info(f"인덱싱 취소됨: {case_id}")
                return progress

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

            # 인덱싱 이력 저장
            _save_indexing_log(progress, "completed")
            return progress

        except Exception as e:
            progress.phase = IndexingPhase.ERROR
            progress.errors.append(str(e))
            progress.completed_at = datetime.now()
            try:
                self.case_store.set_error(case_id, str(e))
            except Exception:
                pass  # 케이스 자체가 없을 수 있음

            # 에러 이력 저장
            _save_indexing_log(progress, "error")
            logger.error(f"인덱싱 파이프라인 에러: {case_id} — {e}")
            return progress

    # ------------------------------------------------------------------
    # 스트리밍 파이프라인: CPU 파싱 → 큐 → GPU 임베딩+저장 동시 실행
    # ------------------------------------------------------------------

    def _run_streaming_pipeline(
        self,
        files: list[Path],
        case_id: str,
        progress: IndexingProgress,
        cancel_flag: threading.Event | None = None,
    ) -> int:
        """CPU 파싱과 GPU 임베딩을 동시에 실행하는 스트리밍 파이프라인

        구조:
            [ProcessPoolExecutor 워커들] ──chunk_queue──→ [GPU 컨슈머 스레드]
            CPU 코어 전체로 파싱+청킹         큐에 쌓이는 청크를 배치로 모아
                                              GPU 임베딩 + ChromaDB 저장

        Returns:
            저장된 총 청크 수
        """
        num_workers = _get_worker_count()
        batch_size = settings.indexing_store_batch_size
        _SENTINEL = None  # 큐 종료 신호

        # 청크 전달 큐 (메모리 제한: 배치 3개분 버퍼)
        chunk_queue: queue.Queue[list[dict[str, Any]] | None] = queue.Queue(
            maxsize=max(3, num_workers)
        )

        # GPU 컨슈머 결과
        consumer_result = {"stored": 0, "errors": []}

        def gpu_consumer() -> None:
            """큐에서 청크를 꺼내 배치로 모아 GPU 임베딩 + 저장"""
            vector_store = self._create_vector_store(case_id)
            buffer: list[Chunk] = []
            batch_num = 0

            while True:
                if cancel_flag and cancel_flag.is_set():
                    break

                try:
                    item = chunk_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is _SENTINEL:
                    # 프로듀서 종료 — 남은 버퍼 플러시
                    if buffer:
                        batch_num += 1
                        _flush_buffer(vector_store, buffer, batch_num, consumer_result)
                        buffer.clear()
                    break

                # dict → Chunk 복원
                for d in item:
                    if "__error__" in d:
                        consumer_result["errors"].append(d["__error__"])
                    else:
                        buffer.append(
                            Chunk(
                                content=d["content"],
                                metadata=d["metadata"],
                                chunk_id=d["chunk_id"],
                                source_type=d["source_type"],
                            )
                        )

                # 버퍼가 배치 크기에 도달하면 GPU로 플러시
                while len(buffer) >= batch_size:
                    batch_num += 1
                    batch = buffer[:batch_size]
                    buffer = buffer[batch_size:]
                    _flush_buffer(vector_store, batch, batch_num, consumer_result)

            # BM25 인덱스 최종 1회 구축
            if consumer_result["stored"] > 0:
                logger.info("BM25 인덱스 구축 시작...")
                vector_store.rebuild_bm25()
                logger.info("BM25 인덱스 구축 완료")

        def _flush_buffer(
            vector_store: Any,
            batch: list[Chunk],
            batch_num: int,
            result: dict[str, Any],
        ) -> None:
            """배치를 GPU 임베딩 + ChromaDB에 저장"""
            try:
                count = vector_store.add_chunks(batch, rebuild_bm25=False)
                result["stored"] += count
                logger.info(
                    f"[GPU] 배치 {batch_num} 저장 완료: {count}개 청크 "
                    f"(누적 {result['stored']}개)"
                )
            except Exception as e:
                error_msg = f"벡터 저장 실패 (batch {batch_num}): {e}"
                result["errors"].append(error_msg)
                logger.warning(error_msg)

        # GPU 컨슈머 스레드 시작
        consumer_thread = threading.Thread(
            target=gpu_consumer, daemon=True, name=f"gpu-consumer-{case_id}"
        )
        consumer_thread.start()

        # CPU 프로듀서: 멀티프로세싱으로 파싱+청킹
        logger.info(
            f"스트리밍 파이프라인 시작: {len(files)}개 파일, "
            f"CPU 워커 {num_workers}개 + GPU 컨슈머 1개"
        )

        args = [(str(f), case_id) for f in files]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_file_worker, arg): arg for arg in args}

            for future in as_completed(futures):
                if cancel_flag and cancel_flag.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    chunk_queue.put(_SENTINEL)
                    consumer_thread.join(timeout=10)
                    return consumer_result["stored"]

                progress.processed_files += 1
                try:
                    result_dicts = future.result()
                    # 청크를 큐에 넣어 GPU 컨슈머로 전달
                    chunk_queue.put(result_dicts)
                except Exception as e:
                    file_str = futures[future][0]
                    error_msg = f"워커 실패 ({Path(file_str).name}): {e}"
                    progress.errors.append(error_msg)
                    logger.warning(error_msg)

                # 진행률 로그 (10% 단위)
                if progress.total_files > 0:
                    step = max(1, progress.total_files // 10)
                    if progress.processed_files % step == 0:
                        logger.info(
                            f"[CPU] 파싱 진행: {progress.processed_files}/{progress.total_files} "
                            f"({progress.progress_percent:.0f}%)"
                        )

        # 프로듀서 완료 — 종료 신호
        chunk_queue.put(_SENTINEL)
        consumer_thread.join()

        # 컨슈머 에러를 progress에 병합
        progress.errors.extend(consumer_result["errors"])

        logger.info(
            f"스트리밍 파이프라인 완료: 파일 {progress.processed_files}개, "
            f"저장 {consumer_result['stored']}개 청크"
        )
        return consumer_result["stored"]

    def _store_chunks_batch(
        self,
        chunks: list[Chunk],
        case_id: str,
        cancel_flag: threading.Event | None = None,
    ) -> int:
        """순차 모드용 벡터 저장 (파일 수가 적을 때)"""
        if not chunks:
            return 0

        vector_store = self._create_vector_store(case_id)
        batch_size = settings.indexing_store_batch_size
        stored_count = 0

        for i in range(0, len(chunks), batch_size):
            if cancel_flag and cancel_flag.is_set():
                return stored_count

            batch = chunks[i : i + batch_size]
            try:
                stored_count += vector_store.add_chunks(batch, rebuild_bm25=False)
            except Exception as e:
                logger.warning(f"벡터 저장 실패: {e}")

        vector_store.rebuild_bm25()
        return stored_count

    def _sequential_process_files(
        self,
        files: list[Path],
        case_id: str,
        progress: IndexingProgress,
        cancel_flag: threading.Event | None = None,
    ) -> list[Chunk]:
        """순차 파싱 (파일 수가 적을 때 또는 fallback)"""
        all_chunks: list[Chunk] = []

        for file_path in files:
            if cancel_flag and cancel_flag.is_set():
                return all_chunks

            try:
                chunks = self._process_file(file_path, case_id, progress)
                all_chunks.extend(chunks)
            except Exception as e:
                error_msg = f"파일 처리 실패 ({file_path.name}): {e}"
                progress.errors.append(error_msg)
                logger.warning(error_msg)
            finally:
                progress.processed_files += 1

        return all_chunks

    @staticmethod
    def _normalize_path(raw: str) -> Path:
        """윈도우 경로 문자열 정규화

        따옴표 제거, 양쪽 공백 제거, 경로 resolve.
        """
        cleaned = raw.strip().strip('"').strip("'").strip()
        return Path(cleaned)

    def _collect_files(self, case_meta: CaseMetadata) -> list[Path]:
        """케이스 데이터 소스에서 처리 대상 파일 수집

        PST 경로: 파일이면 직접 추가, 폴더면 .pst/.ost 재귀 스캔
        문서 경로: 파일이면 직접 추가, 폴더면 지원 확장자 + .pst/.ost 재귀 스캔
        """
        files: list[Path] = []
        pst_extensions = {".pst", ".ost"}
        all_extensions = _DOCUMENT_EXTENSIONS | pst_extensions

        # PST 경로 처리
        for pst_path_str in case_meta.pst_paths:
            pst_path = self._normalize_path(pst_path_str)
            logger.debug(f"PST 경로 확인: '{pst_path_str}' → {pst_path} (exists={pst_path.exists()})")

            if pst_path.is_file():
                if pst_path.suffix.lower() in pst_extensions:
                    files.append(pst_path)
                    logger.info(f"PST 파일 추가: {pst_path.name}")
                else:
                    logger.warning(f"PST 확장자 아님 (무시): {pst_path}")
            elif pst_path.is_dir():
                # 폴더 안의 PST 파일 재귀 스캔
                found = []
                for ext in pst_extensions:
                    found.extend(pst_path.rglob(f"*{ext}"))
                logger.info(f"PST 폴더 스캔: {pst_path} → {len(found)}개 발견")
                files.extend(found)
            else:
                logger.warning(f"PST 경로 없음: {pst_path}")

        # 문서 경로 처리
        for doc_path_str in case_meta.doc_paths:
            doc_path = self._normalize_path(doc_path_str)
            logger.debug(f"문서 경로 확인: '{doc_path_str}' → {doc_path} (exists={doc_path.exists()})")

            if doc_path.is_file():
                if doc_path.suffix.lower() in all_extensions:
                    files.append(doc_path)
                    logger.info(f"문서 파일 추가: {doc_path.name}")
                else:
                    logger.warning(f"지원하지 않는 확장자 (무시): {doc_path}")
            elif doc_path.is_dir():
                found = []
                for ext in all_extensions:
                    found.extend(doc_path.rglob(f"*{ext}"))
                logger.info(f"문서 폴더 스캔: {doc_path} → {len(found)}개 발견")
                files.extend(found)
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

        logger.info(
            f"파일 수집 결과: 총 {len(unique_files)}개 "
            f"(PST 경로 {len(case_meta.pst_paths)}개, 문서 경로 {len(case_meta.doc_paths)}개)"
        )
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


def _save_indexing_log(progress: IndexingProgress, status: str) -> None:
    """인덱싱 이력을 DB에 저장"""
    try:
        import json

        from src.db.database import get_session
        from src.db.models import IndexingLogModel

        elapsed = 0
        if progress.started_at:
            end = progress.completed_at or datetime.now()
            elapsed = int((end - progress.started_at).total_seconds())

        log = IndexingLogModel(
            case_id=progress.case_id,
            started_at=progress.started_at or datetime.now(),
            completed_at=progress.completed_at,
            status=status,
            total_files=progress.total_files,
            processed_files=progress.processed_files,
            total_chunks=progress.total_chunks,
            errors=json.dumps(progress.errors, ensure_ascii=False),
            elapsed_seconds=elapsed,
        )

        with get_session() as session:
            session.add(log)
            session.commit()

    except Exception as e:
        logger.warning(f"인덱싱 이력 저장 실패 (무시): {e}")
