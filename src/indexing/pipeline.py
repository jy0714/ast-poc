"""인덱싱 파이프라인 오케스트레이터

케이스 단위로 PST/문서 파일을 파싱 → 청킹 → 메타데이터 → 임베딩 → 저장하는
전체 Phase A 파이프라인을 관리.

3-stage Producer-Consumer 스트리밍 (`_run_streaming_pipeline`):
    파싱: 파일 크기 1MB 기준 듀얼 풀 분기
        - >= 1MB → ProcessPool (GIL 우회, PyMuPDF/libpff 진짜 병렬)
        - <  1MB → ThreadPool (프로세스 부팅 비용 회피)
    [Process+Thread 파싱] → chunk_queue(20) → [임베딩 워커 1개] → store_queue(10) → [저장 워커 1개]

각 단계가 독립 스레드로 분리되어 GPU/CPU/디스크가 동시에 일하며,
임베딩/저장 진행도를 별도 카운터로 추적 (embedded_chunks vs stored_chunks).
Bounded submission으로 in-flight future 수를 제한해 ProcessPool 메모리 폭주 방지.

파일 수가 적을 때(`num_workers * 2` 이하)는 `_sequential_pipeline` fallback.

진행률은 IndexingProgress 객체로 실시간 추적 가능.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

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


# Windows ProcessPoolExecutor는 max_workers 61 이상에서 에러 발생
_WINDOWS_MAX_WORKERS = 60


def _unload_llm_for_phase_a() -> None:
    """Phase A 시작 시 LLM을 언로드해 임베딩 모델용 VRAM 확보

    Ollama에 keep_alive=0으로 generate 호출을 보내면 모델이 즉시 unload됨.
    LLM이 이미 언로드 상태면 빠르게 no-op 반환. 실패해도 인덱싱은 계속 진행.
    """
    llm_model = settings.ollama_llm_model
    url = f"{settings.ollama_base_url}/api/generate"
    try:
        httpx.post(
            url,
            json={"model": llm_model, "prompt": "", "keep_alive": 0, "stream": False},
            timeout=10.0,
        )
        logger.info(f"Phase A: LLM 언로드 요청 완료 ({llm_model})")
    except Exception as e:
        # 언로드 실패는 치명적이지 않음 — 계속 진행
        logger.warning(f"Phase A: LLM 언로드 실패 (무시): {e}")


# 파싱 풀 분기 임계값 — 1MB 이상이면 ProcessPool, 미만이면 ThreadPool로 라우팅.
# 큰 파일은 PyMuPDF/libpff C-extension이 GIL을 풀더라도 파이썬 측 청킹/메타데이터
# 보강이 누적되어 결국 GIL 직렬화. ProcessPool로 진짜 병렬화.
_PARSE_PROCESS_THRESHOLD_BYTES = 1024 * 1024  # 1MB


def _get_parsing_worker_count() -> int:
    """파싱 풀 워커 수 결정 — min(max_parsing_workers, cpu_count, 60)

    ProcessPool/ThreadPool 각각 이 값을 상한으로 사용 (둘이 별도 풀이지만
    동시 실행되므로 합치면 max 2N개). cpu_count 상한으로 과도한 컨텍스트 스위치 방지.
    """
    base = min(settings.max_parsing_workers, os.cpu_count() or 4)
    return max(1, min(base, _WINDOWS_MAX_WORKERS))


# ThreadPool 워커가 사용할 thread-local DocumentParser — 스레드당 1회 생성
_thread_local = threading.local()


def _get_thread_doc_parser() -> DocumentParser:
    """현재 스레드 전용 DocumentParser — 스레드당 1회 lazy 생성"""
    parser = getattr(_thread_local, "doc_parser", None)
    if parser is None:
        parser = DocumentParser()
        _thread_local.doc_parser = parser
    return parser


def _process_file_thread(file_path: Path, case_id: str) -> tuple[list[Chunk], str | None]:
    """ThreadPool 워커 (작은 파일): 파싱 → 청킹 → 메타데이터 보강

    같은 프로세스 내 스레드이므로 Chunk를 dict로 직렬화할 필요 없이 그대로 반환.
    DocumentParser는 thread-local로 스레드당 1회 생성하여 재사용.

    Returns:
        (chunks, error_msg) — 성공 시 (chunks, None), 실패 시 ([], "에러 메시지")
    """
    try:
        suffix = file_path.suffix.lower()
        doc_parser = _get_thread_doc_parser()

        if suffix in (".pst", ".ost"):
            chunks = _process_pst_standalone(file_path, case_id, doc_parser)
        else:
            chunks = _process_document_standalone(file_path, case_id, doc_parser)

        if chunks:
            enrich_chunks(chunks, case_id=case_id)
        return chunks, None
    except Exception as e:
        return [], f"파일 처리 실패 ({file_path.name}): {e}"


# ProcessPool 워커 프로세스의 모듈 전역 — initializer로 1회 생성, 모든 task에서 재사용
_subprocess_doc_parser: DocumentParser | None = None


def _subprocess_init() -> None:
    """ProcessPoolExecutor 워커 초기화 — DocumentParser 1회 생성

    파일마다 DocumentParser()를 새로 만들면 PyMuPDF/python-docx/Tesseract
    초기화 비용이 누적됨. 워커 프로세스당 1회만 만들어 재사용.
    """
    global _subprocess_doc_parser
    _subprocess_doc_parser = DocumentParser()


def _process_file_subprocess(args: tuple[str, str]) -> tuple[list[Chunk], str | None]:
    """ProcessPool 워커 (큰 파일): 파싱 → 청킹 → 메타데이터 보강

    별도 프로세스에서 실행되므로 결과(Chunk dataclass)는 자동으로 pickle되어 부모로 전송됨.

    Args:
        args: (file_path_str, case_id) — Path는 pickle 가능하지만 args 직렬화 일관성 위해 str

    Returns:
        (chunks, error_msg) — 성공 시 (chunks, None), 실패 시 ([], "에러 메시지")
    """
    file_path_str, case_id = args
    file_path = Path(file_path_str)
    try:
        suffix = file_path.suffix.lower()
        global _subprocess_doc_parser
        if _subprocess_doc_parser is None:
            # initializer 실패/미실행 시 fallback
            _subprocess_doc_parser = DocumentParser()
        doc_parser = _subprocess_doc_parser

        if suffix in (".pst", ".ost"):
            chunks = _process_pst_standalone(file_path, case_id, doc_parser)
        else:
            chunks = _process_document_standalone(file_path, case_id, doc_parser)

        if chunks:
            enrich_chunks(chunks, case_id=case_id)
        return chunks, None
    except Exception as e:
        return [], f"파일 처리 실패 ({file_path.name}): {e}"


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
    parsed_chunks: int = 0  # 파싱→chunk_queue로 전달된 누적 청크 수
    embedded_chunks: int = 0  # GPU 임베딩 완료된 청크 수 (저장 큐로 넘어간 청크)
    stored_chunks: int = 0  # ChromaDB 저장 완료된 청크 수
    errors: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    phase_times: dict[str, float] = field(default_factory=dict)  # 단계별 누적 초
    # 3-stage 파이프라인의 단계별 wall-clock 측정 (workers는 동시 실행 →
    # phase_times 만으론 정확하지 않으므로 워커별 직접 측정)
    parse_wall_seconds: float = 0.0
    embed_wall_seconds: float = 0.0
    store_wall_seconds: float = 0.0
    peak_embed_chunks_per_min: float = 0.0  # 10초 슬라이딩 윈도우 기준 피크 throughput
    last_success_at: float | None = None  # 마지막 임베딩 성공 시각 (epoch). stalled 판단용.
    # 운영 가시성 — 본문 추출 불가능했던 PDF 목록 (파일명, dedup)
    encrypted_pdfs: list[str] = field(default_factory=list)
    scan_pdfs_no_ocr: list[str] = field(default_factory=list)
    _phase_start: float = field(default=0.0, repr=False)

    @property
    def stalled(self) -> bool:
        """마지막 성공 후 indexing_stall_threshold_sec 경과하면 True

        embed가 모두 실패해서 진척 없는 상태를 운영자가 progress API로 감지 가능.
        """
        if not self.is_running:
            return False
        if self.last_success_at is None:
            # 아직 한 번도 성공 못함 + 시작한 지 threshold 경과
            if self.started_at is None:
                return False
            elapsed = (datetime.now() - self.started_at).total_seconds()
            return elapsed >= settings.indexing_stall_threshold_sec
        return (time.time() - self.last_success_at) >= settings.indexing_stall_threshold_sec

    def set_phase(self, phase: IndexingPhase) -> None:
        """단계 전환 + 이전 단계 소요시간 기록"""
        now = time.time()
        if self._phase_start > 0 and self.phase != IndexingPhase.IDLE:
            key = self.phase.value
            self.phase_times[key] = self.phase_times.get(key, 0.0) + (now - self._phase_start)
        self.phase = phase
        self._phase_start = now

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

    @property
    def elapsed_seconds(self) -> float:
        """경과 시간 (초)"""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def files_per_second(self) -> float:
        """파일 처리 속도 (파일/초)"""
        elapsed = self.elapsed_seconds
        if elapsed <= 0 or self.processed_files == 0:
            return 0.0
        return self.processed_files / elapsed

    @property
    def chunks_per_second(self) -> float:
        """청크 저장 속도 (청크/초)"""
        elapsed = self.elapsed_seconds
        if elapsed <= 0 or self.stored_chunks == 0:
            return 0.0
        return self.stored_chunks / elapsed

    @property
    def avg_embed_chunks_per_min(self) -> float:
        """임베딩 평균 throughput (청크/분, 시작부터 현재까지)"""
        elapsed = self.elapsed_seconds
        if elapsed <= 0 or self.embedded_chunks == 0:
            return 0.0
        return self.embedded_chunks * 60.0 / elapsed

    @property
    def eta_seconds(self) -> float | None:
        """예상 남은 시간 (초), 추정 불가하면 None

        파일 처리 속도 기반 (파싱 단계가 보통 가장 오래 걸림). 파일 메타가 없으면
        임베딩 throughput으로 fallback.
        """
        if self.total_files > 0 and self.processed_files > 0:
            remaining = self.total_files - self.processed_files
            if remaining <= 0:
                return 0.0
            return remaining / self.files_per_second if self.files_per_second > 0 else None

        # 파일 메타 없이 청크 단위 추정 (총량을 알 수 없으면 None)
        if self.total_chunks > 0 and self.embedded_chunks > 0:
            remaining_chunks = self.total_chunks - self.embedded_chunks
            if remaining_chunks <= 0:
                return 0.0
            rate_per_sec = self.avg_embed_chunks_per_min / 60.0
            return remaining_chunks / rate_per_sec if rate_per_sec > 0 else None
        return None

    def to_dict(self) -> dict[str, Any]:
        """API 응답용 딕셔너리"""
        elapsed_secs = self.elapsed_seconds
        mins, secs = divmod(int(elapsed_secs), 60)
        hours, mins = divmod(mins, 60)
        if hours:
            elapsed = f"{hours}시간 {mins}분 {secs}초"
        elif mins:
            elapsed = f"{mins}분 {secs}초"
        else:
            elapsed = f"{secs}초"

        eta = self.eta_seconds
        eta_str = ""
        if eta is not None and eta > 0:
            eta_m, eta_s = divmod(int(eta), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            if eta_h:
                eta_str = f"~{eta_h}시간 {eta_m}분"
            elif eta_m:
                eta_str = f"~{eta_m}분 {eta_s}초"
            else:
                eta_str = f"~{eta_s}초"

        return {
            "case_id": self.case_id,
            "status": self.phase.value,
            "phase": self.phase.value,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "total_chunks": self.total_chunks,
            "parsed_chunks": self.parsed_chunks,
            "embedded_chunks": self.embedded_chunks,
            "stored_chunks": self.stored_chunks,
            # 요청서 호환 alias
            "chunks_parsed": self.parsed_chunks,
            "chunks_embedded": self.embedded_chunks,
            "chunks_stored": self.stored_chunks,
            "progress_percent": round(self.progress_percent, 1),
            "elapsed": elapsed,
            "elapsed_seconds": round(elapsed_secs, 1),
            "eta": eta_str,
            "estimated_remaining_seconds": (
                round(eta, 1) if eta is not None else None
            ),
            "files_per_second": round(self.files_per_second, 1),
            "chunks_per_second": round(self.chunks_per_second, 1),
            "throughput_per_min": round(self.avg_embed_chunks_per_min, 1),
            "peak_throughput_per_min": round(self.peak_embed_chunks_per_min, 1),
            "phase_times": {
                k: round(v, 1) for k, v in self.phase_times.items()
            },
            "stage_wall_seconds": {
                "parse": round(self.parse_wall_seconds, 1),
                "embed": round(self.embed_wall_seconds, 1),
                "store": round(self.store_wall_seconds, 1),
            },
            "last_success_at": (
                datetime.fromtimestamp(self.last_success_at).isoformat()
                if self.last_success_at else None
            ),
            "stalled": self.stalled,
            "encrypted_pdfs": list(self.encrypted_pdfs),
            "scan_pdfs_no_ocr": list(self.scan_pdfs_no_ocr),
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
        """케이스별 VectorStoreService 생성 (단일 공유 컬렉션 + case_id 필터)"""
        if self._vector_store_factory:
            return self._vector_store_factory(case_id)

        from src.vectorstore.vector_store import VectorStoreService

        return VectorStoreService(case_id=case_id)

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

    def cancel_and_wait(self, case_id: str, timeout: float | None = None) -> bool:
        """인덱싱 중단 + 워커 스레드 종료 대기

        삭제·재시작 등 즉시 후속 동작이 필요한 호출자가 사용. cancel()과 달리
        워커 스레드가 cancel_flag를 인지하고 정리(파일 핸들 close, vectorstore
        flush 등)할 때까지 join으로 기다림. timeout 안에 종료되지 않아도 워커는
        백그라운드에서 cancel_flag를 계속 체크하므로 곧 멈추지만, 정합성을 위해
        호출자는 timeout 후 별도 정리 작업이 필요할 수 있음.

        Args:
            case_id: 케이스 ID
            timeout: 종료 대기 최대 시간 (초). None이면 settings.indexing_shutdown_timeout.

        Returns:
            True: 실행 중이 아니거나 timeout 안에 정상 종료, False: timeout 초과
        """
        if timeout is None:
            timeout = float(settings.indexing_shutdown_timeout)

        if not self.is_running(case_id):
            return True

        self.cancel(case_id)
        thread = self._threads.get(case_id)
        if thread is None or not thread.is_alive():
            return True

        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                f"인덱싱 종료 대기 timeout ({timeout}초): {case_id} — "
                "워커가 백그라운드에서 cancel을 계속 처리합니다."
            )
            return False
        logger.info(f"인덱싱 워커 종료 확인: {case_id}")
        return True

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
                self._progress[case_id].set_phase(IndexingPhase.ERROR)
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
            if case_meta is None:
                # case_store.get은 보통 raise하지만 방어적으로 가드
                raise ValueError(f"케이스를 찾을 수 없습니다: {case_id}")
            self.case_store.update_status(case_id, CaseStatus.INDEXING)

            # Phase A 시작: LLM 언로드해서 임베딩 모델용 VRAM 확보
            _unload_llm_for_phase_a()

            # 1. 파일 수집
            progress.set_phase(IndexingPhase.SCANNING)
            files = self._collect_files(case_meta)
            progress.total_files = len(files)
            logger.info(f"파일 수집 완료: {len(files)}개 ({case_id})")

            if not files:
                progress.set_phase(IndexingPhase.COMPLETED)
                progress.completed_at = datetime.now()
                self.case_store.update_status(case_id, CaseStatus.READY)
                self.case_store.update_stats(case_id, total_documents=0, total_chunks=0)
                return progress

            # 2~5. 스트리밍 파이프라인 (3-stage: 파싱 / 임베딩 / 저장 분리)
            num_workers = _get_parsing_worker_count()
            use_streaming = len(files) > num_workers * 2 and num_workers > 1

            if use_streaming:
                progress.set_phase(IndexingPhase.PARSING)
                stored_count = self._run_streaming_pipeline(
                    files, case_id, progress, cancel_flag,
                )
            else:
                # 파일 수가 적으면 순차 처리 (배치 즉시 저장)
                progress.set_phase(IndexingPhase.PARSING)
                stored_count = self._sequential_pipeline(
                    files, case_id, progress, cancel_flag,
                )

            if cancel_flag and cancel_flag.is_set():
                progress.set_phase(IndexingPhase.CANCELLED)
                progress.completed_at = datetime.now()
                self.case_store.update_status(case_id, CaseStatus.CREATED)
                _save_indexing_log(progress, "cancelled")
                logger.info(f"인덱싱 취소됨: {case_id}")
                return progress

            progress.total_chunks = stored_count
            progress.stored_chunks = stored_count
            progress.set_phase(IndexingPhase.COMPLETED)
            progress.completed_at = datetime.now()

            # 케이스 통계 업데이트 + 상태 변경
            self.case_store.update_stats(
                case_id,
                total_documents=progress.processed_files,
                total_chunks=stored_count,
            )
            self.case_store.update_status(case_id, CaseStatus.READY)

            summary = (
                f"인덱싱 완료: {case_id} — "
                f"파일 {progress.processed_files}개, 청크 {stored_count}개, "
                f"에러 {len(progress.errors)}개"
            )
            if progress.encrypted_pdfs:
                summary += (
                    f", 암호화 PDF {len(progress.encrypted_pdfs)}개 "
                    f"({', '.join(progress.encrypted_pdfs[:3])}"
                    f"{'...' if len(progress.encrypted_pdfs) > 3 else ''})"
                )
            if progress.scan_pdfs_no_ocr:
                summary += (
                    f", 스캔 PDF(OCR 미적용) {len(progress.scan_pdfs_no_ocr)}개 "
                    f"({', '.join(progress.scan_pdfs_no_ocr[:3])}"
                    f"{'...' if len(progress.scan_pdfs_no_ocr) > 3 else ''})"
                )
            logger.info(summary)

            # 인덱싱 이력 저장
            _save_indexing_log(progress, "completed")
            return progress

        except Exception as e:
            progress.set_phase(IndexingPhase.ERROR)
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
    # 3-stage 스트리밍 파이프라인:
    #   ThreadPool 파싱 → chunk_queue(20) → 임베딩 워커 → store_queue(10) → 저장 워커
    #   CPU(스레드)        버퍼               GPU(단일)          버퍼            디스크 I/O(단일)
    # ------------------------------------------------------------------

    def _run_streaming_pipeline(
        self,
        files: list[Path],
        case_id: str,
        progress: IndexingProgress,
        cancel_flag: threading.Event | None = None,
    ) -> int:
        """3-stage Producer-Consumer 스트리밍 파이프라인 (파싱 / 임베딩 / 저장 분리)

        파싱은 파일 크기로 듀얼 풀 분기:
            - >= 1MB: ProcessPoolExecutor (PyMuPDF/libpff 진짜 병렬, GIL 우회)
            - <  1MB: ThreadPoolExecutor (프로세스 부팅 비용 회피)
        Bounded submission으로 in-flight future를 max_pending개로 제한 →
        ProcessPool 결과 큐에 거대 청크가 무한 누적되는 메모리 폭주 방지.

        구조:
            [ProcessPool: 큰 파일] ┐
                                   ├─→ chunk_queue(20) → [임베딩 1개] → store_queue(10) → [저장 1개]
            [ThreadPool: 작은 파일] ┘     CPU 버퍼          GPU 단일                       디스크 I/O 단일

        Returns:
            저장된 총 청크 수 (ChromaDB upsert 성공 기준)
        """
        num_workers = _get_parsing_worker_count()
        embed_batch_size = settings.embed_batch_size
        embed_concurrent = max(1, settings.embed_max_concurrent)
        # super-batch: embed_batch_size × embed_max_concurrent 개를 모아서 한 번에
        # embed_texts_async 호출 → 내부에서 batch_size 단위로 분할 + Semaphore로
        # embed_concurrent개 동시 전송. Ollama OLLAMA_NUM_PARALLEL≥2 일 때 GPU 활용도 상승.
        super_batch_size = embed_batch_size * embed_concurrent
        _SENTINEL = None

        chunk_queue: queue.Queue[list[Chunk] | None] = queue.Queue(maxsize=20)
        store_queue: queue.Queue[
            tuple[list[Chunk], list[list[float]]] | None
        ] = queue.Queue(maxsize=10)

        # vector_store는 임베딩/저장 워커가 공유 (ChromaDB 클라이언트는 thread-safe)
        vector_store = self._create_vector_store(case_id)
        embedding_service = vector_store._embedding_service

        # thread-safe 카운터/누적 컨테이너 보호
        counter_lock = threading.Lock()

        # 결과 누적 (counter_lock으로 보호)
        state: dict[str, Any] = {
            "embedded": 0,
            "stored": 0,
            "embed_errors": [],
            "store_errors": [],
            "bm25_corpus": [],
            "bm25_ids": [],
            "bm25_metadatas": [],
        }
        # 워커별 wall-clock 측정 (3-stage가 동시 실행되므로 phase_times로는 부정확)
        wall_times: dict[str, float] = {"parse": 0.0, "embed": 0.0, "store": 0.0}
        wall_times_lock = threading.Lock()

        def _save_failed_or_log(chunks: list[Chunk], stage: str, exc: Exception) -> str:
            """배치 전체 실패 시 retry 큐로 보존하고 적절한 에러 메시지 생성"""
            try:
                vector_store._save_failed_chunks(chunks)
                msg = f"{stage} 실패 ({len(chunks)}개 → retry 큐 저장): {exc}"
                logger.warning(msg)
            except Exception as save_e:
                msg = (
                    f"{stage} 실패 + retry 저장 실패 (DATA LOSS {len(chunks)}개): "
                    f"{exc} / {save_e}"
                )
                logger.error(msg, exc_info=True)
            return msg

        def embedding_worker() -> None:
            """chunk_queue → super-batch 수집 → embed_texts_async (동시 전송) → store_queue"""
            buffer: list[Chunk] = []
            batch_num = 0
            worker_active_seconds = 0.0

            def flush_super_batch(batch: list[Chunk]) -> None:
                nonlocal batch_num, worker_active_seconds
                if not batch:
                    return
                batch_num += 1
                num_sub = (len(batch) + embed_batch_size - 1) // embed_batch_size
                t_start = time.monotonic()
                try:
                    # embed_texts (동기 래퍼) → 내부에서 embed_texts_async 호출.
                    # super-batch가 embed_batch_size를 초과하면 자동으로 batch_size
                    # 단위로 분할 + asyncio.Semaphore(embed_concurrent)로 동시 전송.
                    # cancel_flag를 전달 → 임베딩 재시도/분할 루프에서도 즉시 중단됨.
                    embeddings = embedding_service.embed_texts(
                        [c.content for c in batch],
                        cancel_event=cancel_flag,
                    )
                    elapsed = time.monotonic() - t_start
                    worker_active_seconds += elapsed

                    failed_idx_set = set(embedding_service._last_failed_indices)
                    if failed_idx_set:
                        sorted_failed = sorted(failed_idx_set)
                        failed_chunks = [batch[i] for i in sorted_failed]
                        failed_diag = [
                            embedding_service._last_failed_diagnostics.get(i, {})
                            for i in sorted_failed
                        ]
                        try:
                            vector_store._save_failed_chunks(
                                failed_chunks, diagnostics=failed_diag
                            )
                            err_msg = (
                                f"임베딩 super-batch {batch_num} 부분 실패: "
                                f"{len(failed_idx_set)}/{len(batch)}개 (retry 큐 저장)"
                            )
                            logger.warning(err_msg)
                        except Exception as save_e:
                            err_msg = (
                                f"임베딩 super-batch {batch_num} 부분 실패 + retry 저장 실패 "
                                f"(DATA LOSS {len(failed_idx_set)}개): {save_e}"
                            )
                            logger.error(err_msg, exc_info=True)
                        with counter_lock:
                            state["embed_errors"].append(err_msg)
                    # progress.last_success_at 갱신 (성공한 청크가 있으면)
                    if len(failed_idx_set) < len(batch):
                        with counter_lock:
                            progress.last_success_at = (
                                embedding_service._last_success_at or time.time()
                            )

                    kept_indices = [
                        i for i in range(len(batch)) if i not in failed_idx_set
                    ]
                    if not kept_indices:
                        return
                    kept_chunks = [batch[i] for i in kept_indices]
                    kept_embeddings = [embeddings[i] for i in kept_indices]
                    with counter_lock:
                        state["embedded"] += len(kept_chunks)
                        progress.embedded_chunks = state["embedded"]
                    throughput = (
                        len(kept_chunks) * 60.0 / elapsed if elapsed > 0 else 0.0
                    )
                    logger.info(
                        f"[EMBED] super-batch {batch_num} 완료: "
                        f"{len(kept_chunks)}개 ({num_sub}개 sub-batch × "
                        f"{embed_concurrent} 동시), {elapsed:.2f}초, "
                        f"{throughput:.0f} chunks/min "
                        f"(누적 {state['embedded']}개)"
                    )
                    store_queue.put((kept_chunks, kept_embeddings))
                except Exception as e:
                    worker_active_seconds += time.monotonic() - t_start
                    err_msg = _save_failed_or_log(
                        batch, f"임베딩 (super-batch {batch_num})", e
                    )
                    with counter_lock:
                        state["embed_errors"].append(err_msg)

            while True:
                if cancel_flag and cancel_flag.is_set():
                    break
                try:
                    item = chunk_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if item is _SENTINEL:
                    flush_super_batch(buffer)
                    buffer = []
                    break
                buffer.extend(item)
                # super-batch 크기 도달 시 한 번에 전송
                while len(buffer) >= super_batch_size:
                    if cancel_flag and cancel_flag.is_set():
                        break
                    batch = buffer[:super_batch_size]
                    buffer = buffer[super_batch_size:]
                    flush_super_batch(batch)

            with wall_times_lock:
                wall_times["embed"] = worker_active_seconds
                progress.embed_wall_seconds = worker_active_seconds
            store_queue.put(_SENTINEL)

        def storage_worker() -> None:
            """store_queue에서 (chunks, embeddings)를 꺼내 ChromaDB upsert + BM25 corpus 누적"""
            worker_active_seconds = 0.0
            while True:
                if cancel_flag and cancel_flag.is_set():
                    break
                try:
                    item = store_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if item is _SENTINEL:
                    break
                chunks, embeddings = item
                t_start = time.monotonic()
                try:
                    count = vector_store.add_chunks_with_embeddings(
                        chunks, embeddings, rebuild_bm25=False
                    )
                    elapsed = time.monotonic() - t_start
                    worker_active_seconds += elapsed
                    with counter_lock:
                        state["stored"] += count
                        progress.stored_chunks = state["stored"]
                        for c in chunks:
                            state["bm25_corpus"].append(c.content)
                            state["bm25_ids"].append(c.chunk_id)
                            meta = dict(c.metadata) if c.metadata else {}
                            meta["source_type"] = c.source_type
                            if vector_store.case_id:
                                meta["case_id"] = vector_store.case_id
                            state["bm25_metadatas"].append(meta)
                    logger.info(
                        f"[STORE] 저장 완료: {count}개, {elapsed:.2f}초 "
                        f"(누적 {state['stored']}개, "
                        f"{progress.chunks_per_second:.1f} 청크/초)"
                    )
                except Exception as e:
                    worker_active_seconds += time.monotonic() - t_start
                    err_msg = _save_failed_or_log(chunks, "저장", e)
                    with counter_lock:
                        state["store_errors"].append(err_msg)
            with wall_times_lock:
                wall_times["store"] = worker_active_seconds
                progress.store_wall_seconds = worker_active_seconds

        # 10초마다 throughput을 INFO 로그로 보고하는 모니터 스레드
        monitor_stop = threading.Event()

        def throughput_monitor() -> None:
            """10초 슬라이딩 윈도우로 임베딩 throughput을 측정 + ETA 출력"""
            interval = 10.0
            last_embedded = 0
            last_t = time.monotonic()
            while not monitor_stop.wait(interval):
                now = time.monotonic()
                with counter_lock:
                    current = state["embedded"]
                    total = progress.total_chunks
                    parsed = progress.parsed_chunks
                delta = current - last_embedded
                dt = now - last_t
                if dt <= 0:
                    continue
                rate_per_min = delta * 60.0 / dt
                if rate_per_min > progress.peak_embed_chunks_per_min:
                    progress.peak_embed_chunks_per_min = rate_per_min

                # ETA: 총 청크 수를 모르면 파싱된 청크 기준으로 추정 (under-count)
                target = total if total > 0 else parsed
                remaining = max(0, target - current)
                eta_text = ""
                if rate_per_min > 0 and remaining > 0:
                    eta_min = remaining / rate_per_min
                    if eta_min >= 60:
                        eta_text = f", 예상 잔여 약 {eta_min / 60:.1f}시간"
                    elif eta_min >= 1:
                        eta_text = f", 예상 잔여 약 {eta_min:.0f}분"
                    else:
                        eta_text = f", 예상 잔여 약 {eta_min * 60:.0f}초"
                target_text = (
                    f"누적 {current:,}/{target:,}" if target > 0 else f"누적 {current:,}"
                )
                logger.info(
                    f"[THROUGHPUT] 임베딩: {rate_per_min:.0f} chunks/min "
                    f"({target_text}{eta_text})"
                )
                last_embedded = current
                last_t = now

        embed_thread = threading.Thread(
            target=embedding_worker, daemon=True, name=f"embed-{case_id}"
        )
        store_thread = threading.Thread(
            target=storage_worker, daemon=True, name=f"store-{case_id}"
        )
        monitor_thread = threading.Thread(
            target=throughput_monitor, daemon=True, name=f"monitor-{case_id}"
        )
        embed_thread.start()
        store_thread.start()
        monitor_thread.start()

        # 파일을 크기로 분류: 큰 파일은 ProcessPool(GIL 우회), 작은 파일은 ThreadPool
        # (프로세스 부팅 비용 회피). 양쪽 모두 크기 내림차순으로 큰 것부터 시작.
        large_files: list[Path] = []
        small_files: list[Path] = []
        sized_files: list[tuple[Path, int]] = []
        for f in files:
            try:
                size = f.stat().st_size if f.exists() else 0
            except OSError:
                size = 0
            sized_files.append((f, size))
        sized_files.sort(key=lambda x: x[1], reverse=True)
        for f, size in sized_files:
            if size >= _PARSE_PROCESS_THRESHOLD_BYTES:
                large_files.append(f)
            else:
                small_files.append(f)

        # 풀별 워커 수 — 대상 파일이 없으면 풀 자체를 만들지 않음 (cold-start 비용 회피)
        process_workers = min(num_workers, len(large_files)) if large_files else 0
        thread_workers = min(num_workers, len(small_files)) if small_files else 0

        logger.info(
            f"3-stage 스트리밍 파이프라인 시작: {len(files)}개 파일 "
            f"(>=1MB ProcessPool {len(large_files)}개 / <1MB ThreadPool {len(small_files)}개), "
            f"임베딩 1개 (super-batch={super_batch_size}={embed_batch_size}×{embed_concurrent} 동시) + 저장 1개 "
            f"(chunk_queue={chunk_queue.maxsize}, store_queue={store_queue.maxsize})"
        )

        # 파싱 풀별 카운터 — 완료 후 요약 로그용
        parse_summary = {
            "process_success": 0,
            "process_fail": 0,
            "thread_success": 0,
            "thread_fail": 0,
        }
        parse_start_t = time.monotonic()

        try:
            with ExitStack() as stack:
                proc_exec: ProcessPoolExecutor | None = None
                thread_exec: ThreadPoolExecutor | None = None
                if process_workers > 0:
                    proc_exec = stack.enter_context(
                        ProcessPoolExecutor(
                            max_workers=process_workers,
                            initializer=_subprocess_init,
                        )
                    )
                if thread_workers > 0:
                    thread_exec = stack.enter_context(
                        ThreadPoolExecutor(
                            max_workers=thread_workers,
                            thread_name_prefix=f"parse-{case_id}",
                        )
                    )

                # Bounded submission — 한 번에 최대 max_pending개의 future만 in-flight.
                # ProcessPool 결과 큐에 거대 청크가 무한 누적되는 것 방지.
                # chunk_queue.put이 block되면 자연스럽게 backfill도 멈춤 → 진짜 backpressure.
                large_iter = iter(large_files)
                small_iter = iter(small_files)
                pending: dict[Future, tuple[Path, str]] = {}
                max_pending = max(num_workers * 2, 4)

                def _submit_next() -> bool:
                    """대기 중인 파일 1개를 적절한 풀에 submit. 더 없으면 False."""
                    if proc_exec is not None:
                        for f in large_iter:
                            fut = proc_exec.submit(
                                _process_file_subprocess, (str(f), case_id)
                            )
                            pending[fut] = (f, "process")
                            return True
                    if thread_exec is not None:
                        for f in small_iter:
                            fut = thread_exec.submit(
                                _process_file_thread, f, case_id
                            )
                            pending[fut] = (f, "thread")
                            return True
                    return False

                # Prime
                for _ in range(max_pending):
                    if not _submit_next():
                        break

                cancelled = False
                while pending:
                    if cancel_flag and cancel_flag.is_set():
                        cancelled = True
                        if proc_exec is not None:
                            proc_exec.shutdown(wait=False, cancel_futures=True)
                        if thread_exec is not None:
                            thread_exec.shutdown(wait=False, cancel_futures=True)
                        break

                    done, _not_done = wait(
                        list(pending), return_when=FIRST_COMPLETED, timeout=1.0
                    )
                    if not done:
                        continue

                    for future in done:
                        file_path, mode = pending.pop(future)
                        try:
                            chunks, err = future.result()
                            if err:
                                with counter_lock:
                                    progress.errors.append(err)
                                parse_summary[f"{mode}_fail"] += 1
                                logger.warning(err)
                            else:
                                if chunks:
                                    # chunk_queue가 가득 차면 여기서 block → 자연스러운 backpressure
                                    chunk_queue.put(chunks)
                                    with counter_lock:
                                        progress.parsed_chunks += len(chunks)
                                        # 본문 추출 불가 PDF 추적 (multi-process 경로)
                                        meta0 = chunks[0].metadata
                                        if meta0.get("file_type") == "pdf":
                                            fname = meta0.get("filename", file_path.name)
                                            if (meta0.get("is_encrypted")
                                                and fname not in progress.encrypted_pdfs):
                                                progress.encrypted_pdfs.append(fname)
                                            if (meta0.get("scan_pdf")
                                                and fname not in progress.scan_pdfs_no_ocr):
                                                progress.scan_pdfs_no_ocr.append(fname)
                                parse_summary[f"{mode}_success"] += 1
                        except Exception as e:
                            error_msg = (
                                f"파싱 워커 실패 ({file_path.name}, {mode}): {e}"
                            )
                            with counter_lock:
                                progress.errors.append(error_msg)
                            parse_summary[f"{mode}_fail"] += 1
                            logger.warning(error_msg)
                        finally:
                            with counter_lock:
                                progress.processed_files += 1

                        # 진행률 로그 (10% 단위)
                        if progress.total_files > 0:
                            step = max(1, progress.total_files // 10)
                            if progress.processed_files % step == 0:
                                eta = progress.eta_seconds
                                eta_str = f", ETA {int(eta)}초" if eta else ""
                                logger.info(
                                    f"[PARSE] 파싱 진행: "
                                    f"{progress.processed_files}/"
                                    f"{progress.total_files} "
                                    f"({progress.progress_percent:.0f}%, "
                                    f"{progress.files_per_second:.1f} 파일/초"
                                    f"{eta_str})"
                                )

                        # backfill — 다음 파일 1개 submit
                        _submit_next()
        finally:
            # 파싱 wall-clock 종료 (제출 완료 시점까지)
            parse_elapsed = time.monotonic() - parse_start_t
            with wall_times_lock:
                wall_times["parse"] = parse_elapsed
                progress.parse_wall_seconds = parse_elapsed
            # 파싱 종료 신호 → 임베딩 워커가 store_queue로 sentinel 전파
            chunk_queue.put(_SENTINEL)
            embed_thread.join()
            store_thread.join()
            monitor_stop.set()
            monitor_thread.join(timeout=2.0)

        # 파싱 단계 요약 — 풀별 성공/실패
        logger.info(
            f"[PARSE] 완료: 성공 {parse_summary['process_success'] + parse_summary['thread_success']}개 "
            f"(process {parse_summary['process_success']}, thread {parse_summary['thread_success']}), "
            f"실패 {parse_summary['process_fail'] + parse_summary['thread_fail']}개 "
            f"(process {parse_summary['process_fail']}, thread {parse_summary['thread_fail']})"
            + (" — CANCELLED" if cancelled else "")
        )

        # 워커 에러를 progress에 병합
        progress.errors.extend(state["embed_errors"])
        progress.errors.extend(state["store_errors"])

        # BM25 인덱스 최종 1회 구축 (저장 성공한 청크만 포함 → 벡터/키워드 일관성)
        if state["stored"] > 0:
            logger.info("BM25 인덱스 구축 시작...")
            vector_store.rebuild_bm25(
                corpus=state["bm25_corpus"],
                ids=state["bm25_ids"],
                metadatas=state["bm25_metadatas"],
            )
            logger.info("BM25 인덱스 구축 완료")

        # 임계 실패율 검사
        total_stage_errors = len(state["embed_errors"]) + len(state["store_errors"])
        _check_failure_rate(
            progress=progress,
            stored=state["stored"],
            store_errors=total_stage_errors,
            total_files=len(files),
        )

        # === 벤치마크 요약 ===
        total_files_processed = parse_summary["process_success"] + parse_summary["thread_success"]
        total_files_failed = parse_summary["process_fail"] + parse_summary["thread_fail"]
        total_elapsed = max(0.001, progress.elapsed_seconds)
        avg_throughput = state["embedded"] * 60.0 / total_elapsed if state["embedded"] else 0.0

        logger.info("=" * 70)
        logger.info("[BENCHMARK] 인덱싱 단계별 소요 시간 요약")
        logger.info(
            f"  파싱     : {wall_times['parse']:>8.1f}초 "
            f"(파일 {len(files)}개, 성공 {total_files_processed}개, 실패 {total_files_failed}개)"
        )
        logger.info(
            f"  임베딩   : {wall_times['embed']:>8.1f}초 "
            f"(super-batch {embed_concurrent}-동시, 청크 {state['embedded']}개)"
        )
        logger.info(
            f"  저장     : {wall_times['store']:>8.1f}초 "
            f"(청크 {state['stored']}개)"
        )
        logger.info(f"  전체     : {total_elapsed:>8.1f}초 (wall-clock)")
        logger.info(f"  총 청크 수      : {state['stored']:,}개")
        logger.info(f"  평균 throughput : {avg_throughput:>8.0f} chunks/min")
        logger.info(
            f"  피크 throughput : {progress.peak_embed_chunks_per_min:>8.0f} chunks/min "
            f"(10초 윈도우)"
        )
        logger.info("=" * 70)

        logger.info(
            f"3-stage 파이프라인 완료: 파일 {progress.processed_files}개, "
            f"임베딩 {state['embedded']}개, 저장 {state['stored']}개"
        )
        return state["stored"]

    def _sequential_pipeline(
        self,
        files: list[Path],
        case_id: str,
        progress: IndexingProgress,
        cancel_flag: threading.Event | None = None,
    ) -> int:
        """순차 파싱 + 배치 즉시 저장 (메모리 절약)

        파일 하나씩 파싱→청킹 후 버퍼에 쌓고, 배치 크기에 도달하면
        즉시 GPU 임베딩+저장. 전체 청크를 메모리에 보관하지 않음.
        """
        vector_store = self._create_vector_store(case_id)
        batch_size = settings.indexing_store_batch_size
        buffer: list[Chunk] = []
        stored_count = 0
        batch_num = 0
        bm25_corpus: list[str] = []
        bm25_ids: list[str] = []
        bm25_metadatas: list[dict[str, Any]] = []

        for file_path in files:
            if cancel_flag and cancel_flag.is_set():
                break

            try:
                chunks = self._process_file(file_path, case_id, progress)
                buffer.extend(chunks)
            except Exception as e:
                error_msg = f"파일 처리 실패 ({file_path.name}): {e}"
                progress.errors.append(error_msg)
                logger.warning(error_msg)
            finally:
                progress.processed_files += 1

            def _flush(batch: list[Chunk], n: int) -> int:
                """순차 파이프라인용 배치 플러시 — 실패 청크는 retry 큐로"""
                try:
                    progress.set_phase(IndexingPhase.STORING)
                    cnt = vector_store.add_chunks(batch, rebuild_bm25=False)
                    failed_idx_set = set(
                        vector_store._embedding_service._last_failed_indices
                    )
                    for i, c in enumerate(batch):
                        if i in failed_idx_set:
                            continue
                        bm25_corpus.append(c.content)
                        bm25_ids.append(c.chunk_id)
                        meta = dict(c.metadata) if c.metadata else {}
                        meta["source_type"] = c.source_type
                        if vector_store.case_id:
                            meta["case_id"] = vector_store.case_id
                        bm25_metadatas.append(meta)
                    if failed_idx_set:
                        logger.warning(
                            f"배치 {n} 부분 실패: {len(failed_idx_set)}/{len(batch)}개 "
                            f"임베딩 실패 (retry 큐 저장됨)"
                        )
                    return cnt
                except Exception as e:
                    try:
                        vector_store._save_failed_chunks(batch)
                        logger.warning(
                            f"벡터 저장 실패 (batch {n}, {len(batch)}개 → retry 큐): {e}"
                        )
                    except Exception as save_e:
                        logger.error(
                            f"벡터 저장 실패 + retry 저장 실패 (batch {n}, "
                            f"DATA LOSS {len(batch)}개): {e} / {save_e}"
                        )
                    return 0
                finally:
                    progress.set_phase(IndexingPhase.PARSING)

            # 버퍼가 배치 크기에 도달하면 즉시 저장
            while len(buffer) >= batch_size:
                if cancel_flag and cancel_flag.is_set():
                    break
                batch_num += 1
                batch = buffer[:batch_size]
                buffer = buffer[batch_size:]
                count = _flush(batch, batch_num)
                stored_count += count
                progress.stored_chunks = stored_count
                progress.total_chunks = stored_count

        # 남은 버퍼 플러시
        if buffer and not (cancel_flag and cancel_flag.is_set()):
            batch_num += 1

            def _flush_final(batch: list[Chunk], n: int) -> int:
                try:
                    progress.set_phase(IndexingPhase.STORING)
                    cnt = vector_store.add_chunks(batch, rebuild_bm25=False)
                    failed_idx_set = set(
                        vector_store._embedding_service._last_failed_indices
                    )
                    for i, c in enumerate(batch):
                        if i in failed_idx_set:
                            continue
                        bm25_corpus.append(c.content)
                        bm25_ids.append(c.chunk_id)
                        meta = dict(c.metadata) if c.metadata else {}
                        meta["source_type"] = c.source_type
                        if vector_store.case_id:
                            meta["case_id"] = vector_store.case_id
                        bm25_metadatas.append(meta)
                    return cnt
                except Exception as e:
                    try:
                        vector_store._save_failed_chunks(batch)
                        logger.warning(
                            f"벡터 저장 실패 (batch {n}, {len(batch)}개 → retry 큐): {e}"
                        )
                    except Exception as save_e:
                        logger.error(
                            f"벡터 저장 실패 + retry 저장 실패 (batch {n}, "
                            f"DATA LOSS {len(batch)}개): {e} / {save_e}"
                        )
                    return 0

            count = _flush_final(buffer, batch_num)
            stored_count += count
            progress.stored_chunks = stored_count

        # BM25 인덱스 최종 구축
        if stored_count > 0:
            vector_store.rebuild_bm25(
                corpus=bm25_corpus,
                ids=bm25_ids,
                metadatas=bm25_metadatas,
            )

        # 임계 실패율 검사
        store_failures = sum(
            1 for e in progress.errors if e.startswith("벡터 저장 실패")
        )
        _check_failure_rate(
            progress=progress,
            stored=stored_count,
            store_errors=store_failures,
            total_files=len(files),
        )

        return stored_count

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

        progress.set_phase(IndexingPhase.PARSING)
        parser = PSTParser(pst_path)
        result = parser.parse()

        all_chunks: list[Chunk] = []
        pst_meta = {"pst_file": pst_path.name}

        # 이메일 청킹
        progress.set_phase(IndexingPhase.CHUNKING)
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
        progress.set_phase(IndexingPhase.ENRICHING)
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
        progress.set_phase(IndexingPhase.PARSING)
        parsed = self._doc_parser.parse(doc_path)

        progress.set_phase(IndexingPhase.CHUNKING)
        # XLSX는 list[ParsedDocument] 반환
        docs = parsed if isinstance(parsed, list) else [parsed]

        # 본문 추출 불가 PDF 추적 (운영 가시성)
        if docs and docs[0].file_type == "pdf":
            meta0 = docs[0].metadata
            if meta0.get("is_encrypted") and doc_path.name not in progress.encrypted_pdfs:
                progress.encrypted_pdfs.append(doc_path.name)
            if meta0.get("scan_pdf") and doc_path.name not in progress.scan_pdfs_no_ocr:
                progress.scan_pdfs_no_ocr.append(doc_path.name)

        all_chunks: list[Chunk] = []
        for doc in docs:
            chunks = self._doc_chunker.chunk_parsed_document(doc)
            all_chunks.extend(chunks)

        # 메타데이터 보강
        progress.set_phase(IndexingPhase.ENRICHING)
        if all_chunks:
            enrich_chunks(all_chunks, case_id=case_id)

        logger.info(f"문서 처리 완료: {doc_path.name} → {len(all_chunks)}개 청크")
        return all_chunks


# 인덱싱 실패 임계값 — 저장 시도 대비 실패 배치/파일 비율
_FAILURE_RATE_THRESHOLD = 0.10  # 10% 초과 시 abort


class IndexingFailureRateExceeded(Exception):
    """저장 실패율이 임계치를 초과해 인덱싱을 중단해야 함"""


def _check_failure_rate(
    progress: IndexingProgress,
    stored: int,
    store_errors: int,
    total_files: int,
) -> None:
    """저장 실패율이 임계치를 넘으면 예외를 발생시켜 파이프라인 abort

    파일 처리 에러는 별도(workers fail). 여기서는 GPU 저장 실패만 본다 —
    저장 단계에서 silent하게 청크가 누락되면 검색 품질 저하로 직결되기 때문.
    """
    # 저장 시도 횟수 추정: 저장된 배치 수 + 실패한 배치 수
    # stored=0이고 errors도 0이면 인덱싱 대상이 없었던 것 (예: 빈 케이스)
    if stored == 0 and store_errors == 0:
        return

    # 배치 단위 실패율 (간단 근사)
    total_attempts = max(1, store_errors + max(1, stored // max(1, settings.indexing_store_batch_size)))
    failure_rate = store_errors / total_attempts

    if failure_rate > _FAILURE_RATE_THRESHOLD:
        msg = (
            f"벡터 저장 실패율 {failure_rate:.1%} > 임계치 "
            f"{_FAILURE_RATE_THRESHOLD:.0%} (실패 {store_errors}, 저장 {stored}). "
            f"silent 데이터 손실 위험으로 인덱싱 중단."
        )
        logger.error(msg)
        raise IndexingFailureRateExceeded(msg)


def _save_indexing_log(progress: IndexingProgress, status: str) -> None:
    """인덱싱 이력을 DB에 저장 — 케이스 부재/DB 미초기화에 대해 명확히 분류

    실패 시나리오와 대응:
    - 케이스가 이미 삭제됨 (FK violation 또는 사전 SELECT NULL): INFO + skip.
      운영 중 흔한 race이고 데이터 손실 아님.
    - indexing_logs 테이블 부재 (no such table): ERROR + skip.
      init_db()가 호출되지 않은 상태이므로 인프라 문제. 다음 호출도 동일 실패 예상.
    - DB 파일 접근 불가 (unable to open database file): ERROR + skip.
      tmpdir cleanup 후 백그라운드 워커가 DB 호출 시 흔히 발생.
    - 그 외: WARNING.
    """
    try:
        import json

        from sqlalchemy.exc import IntegrityError, OperationalError

        from src.db.database import get_session
        from src.db.models import CaseModel, IndexingLogModel

        elapsed = 0
        if progress.started_at:
            end = progress.completed_at or datetime.now()
            elapsed = int((end - progress.started_at).total_seconds())

        # 케이스 존재 사전 확인 — FK violation을 ERROR 로그 없이 INFO로 처리.
        # 인덱싱 도중 사용자가 케이스를 삭제하면 워커가 완료 시 호출되어 흔히 발생.
        with get_session() as session:
            case_exists = session.get(CaseModel, progress.case_id) is not None
            if not case_exists:
                logger.info(
                    f"인덱싱 이력 저장 skip: 케이스 {progress.case_id}가 이미 삭제됨"
                )
                return

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
            try:
                session.add(log)
                session.commit()
            except IntegrityError as ie:
                # 사전 확인과 commit 사이에 케이스가 삭제되는 race
                logger.info(
                    f"인덱싱 이력 저장 skip (FK race): 케이스 {progress.case_id} — {ie}"
                )
                session.rollback()
                return
            except OperationalError as oe:
                # 테이블 부재 / DB 파일 접근 불가 — 인프라 문제
                msg = str(oe).lower()
                if "no such table" in msg or "unable to open database file" in msg:
                    logger.error(
                        f"인덱싱 이력 저장 실패 (인프라 문제, DB init 확인 필요): "
                        f"{progress.case_id} — {oe}"
                    )
                else:
                    logger.warning(
                        f"인덱싱 이력 저장 실패: {progress.case_id} — {oe}"
                    )
                session.rollback()
                return

    except Exception as e:
        logger.warning(f"인덱싱 이력 저장 실패 (무시): {e}")
