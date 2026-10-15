"""인덱싱 파이프라인 오케스트레이터 (v4)

케이스 단위로 PST/문서 파일을 파싱 → 증거 단위 생성 → 임베딩 → 케이스 색인 저장까지
전체 Phase A 파이프라인을 관리한다 (지시서 §5.8).

스트리밍 구조 (`_run_pipeline`):
    파싱+단위 생성: 파일 크기 1MB 기준 듀얼 풀 분기
        - >= 1MB → ProcessPool (GIL 우회, PyMuPDF/libpff 진짜 병렬)
        - <  1MB → ThreadPool (프로세스 부팅 비용 회피)
    [Process+Thread: 파일 → 단위] → (메인: unit_id 중복 제거) → part_queue(20) → [임베딩 워커 1개]
    → 종료 후 케이스 색인 1회 저장(units.jsonl + BM25 + 서비스 프로필 dense.npy, 폴더 원자 교체)

- 같은 메일이 여러 PST에 있으면 같은 unit_id → 처음 것만 남기고 folder_paths 병합(UnitCollector).
- 증분 색인: 기존 색인에 같은 part_id·같은 임베딩 입력이 있으면 벡터를 재사용하고 새 파트만
  임베딩한다. 데이터 소스를 추가해 다시 돌려도 바뀐 부분만 GPU를 쓴다.
- 영구 임베딩 실패 파트는 data/failed_embeddings/{case_id}.jsonl에 보존하고 색인(Dense·BM25 모두)
  에서 뺀다 — 두 검색기의 대상 파트가 같게 유지된다.
- 진행률 필드 이름(parsed_chunks 등)은 프론트엔드 호환을 위해 유지하고 값은 파트 수다.
  `IndexingPhase.CHUNKING`은 "단위 생성" 의미로 쓴다.

진행률은 IndexingProgress 객체로 실시간 추적 가능.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    BrokenExecutor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from src.cases.case_store import CaseMetadata, CaseStatus, CaseStore
from src.index import case_index
from src.index.case_index import ProfileVectors, embed_input, input_hash, profile_spec
from src.parsers.document_parser import DocumentParser
from src.units.builder import BuildStats, UnitBuilder, UnitCollector, relative_source_path
from src.units.schema import EvidenceUnit, UnitPart
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
    if settings.ast_fake_models:
        return
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
_PARSE_PROCESS_THRESHOLD_BYTES = 1024 * 1024  # 1MB


def _get_parsing_worker_count() -> int:
    """파싱 풀 워커 수 결정 — min(max_parsing_workers, cpu_count, 60)

    ProcessPool/ThreadPool 각각 이 값을 상한으로 사용 (둘이 별도 풀이지만
    동시 실행되므로 합치면 max 2N개). cpu_count 상한으로 과도한 컨텍스트 스위치 방지.
    """
    base = min(settings.max_parsing_workers, os.cpu_count() or 4)
    return max(1, min(base, _WINDOWS_MAX_WORKERS))


# 파일 1개 처리 결과: (단위 목록, BuildStats.to_dict(), 에러 메시지 | None)
FileResult = tuple[list[EvidenceUnit], dict[str, Any], str | None]

# 파싱 프로세스가 죽어(BrokenProcessPool) 풀을 다시 만드는 최대 횟수. 넘으면 남은 큰 파일은 ThreadPool로.
_MAX_PROCESS_POOL_RESTARTS = 5
# 풀이 깨질 때 처리 중이던 파일은 다시 시도한다. 이 횟수만큼 깨짐에 연루되면 그 파일을 실패로 둔다.
_MAX_CRASH_INVOLVEMENT = 2

# ThreadPool 워커가 사용할 thread-local UnitBuilder — 스레드·케이스당 1회 생성
_thread_local = threading.local()


def _thread_builder(case_id: str) -> UnitBuilder:
    """현재 스레드 전용 UnitBuilder (DocumentParser 포함)"""
    builders: dict[str, UnitBuilder] | None = getattr(_thread_local, "builders", None)
    if builders is None:
        builders = _thread_local.builders = {}
    if case_id not in builders:
        builders[case_id] = UnitBuilder(case_id, doc_parser=DocumentParser(detect_language=False))
    return builders[case_id]


def _build_file(builder: UnitBuilder, file_path: Path, rel_path: str) -> FileResult:
    """파일 1개 → 증거 단위 (파일별 통계 포함)"""
    builder.stats = BuildStats()
    try:
        units = builder.build_file(file_path, rel_path)
        return units, builder.stats.to_dict(), None
    except Exception as e:
        return [], builder.stats.to_dict(), f"파일 처리 실패 ({file_path.name}): {e}"


def _process_file_thread(file_path: Path, rel_path: str, case_id: str) -> FileResult:
    """ThreadPool 워커 (작은 파일): 파싱 → 단위 생성"""
    return _build_file(_thread_builder(case_id), file_path, rel_path)


# ProcessPool 워커 프로세스의 모듈 전역 — initializer로 1회 생성, 모든 task에서 재사용
_subprocess_builders: dict[str, UnitBuilder] = {}


def _subprocess_init() -> None:
    """ProcessPoolExecutor 워커 초기화 (빌더는 케이스별로 첫 task에서 생성)"""
    _subprocess_builders.clear()


def _process_file_subprocess(args: tuple[str, str, str]) -> FileResult:
    """ProcessPool 워커 (큰 파일): 파싱 → 단위 생성. 결과는 pickle되어 부모로 전송된다.

    Args:
        args: (file_path_str, rel_path, case_id)
    """
    file_path_str, rel_path, case_id = args
    builder = _subprocess_builders.get(case_id)
    if builder is None:
        builder = _subprocess_builders[case_id] = UnitBuilder(
            case_id, doc_parser=DocumentParser(detect_language=False)
        )
    return _build_file(builder, Path(file_path_str), rel_path)


def case_source_files(case_meta: CaseMetadata) -> list[tuple[Path, str]]:
    """케이스 데이터 소스 → [(파일 경로, 케이스 기준 상대 경로)]

    상대 경로는 그 파일을 포함하는 가장 구체적인 데이터 소스 루트 기준이다. 문서 단위 ID가
    이 경로에서 나오므로(지시서 §5.1.2) 인덱싱과 scripts/unit_stats.py가 같은 함수를 쓴다.
    """
    roots = [IndexingPipeline._normalize_path(p) for p in [*case_meta.pst_paths, *case_meta.doc_paths]]
    out: list[tuple[Path, str]] = []
    for f in collect_case_files(case_meta):
        resolved = f.resolve()
        candidates = [r for r in roots if r == f or (r.is_dir() and resolved.is_relative_to(r.resolve()))]
        root = max(candidates, key=lambda r: len(str(r))) if candidates else f.parent
        out.append((f, relative_source_path(f, root)))
    return out


def _file_fingerprint(path: Path, rel: str) -> dict[str, Any]:
    """manifest source_files 항목 (내용 해시 대신 크기·수정 시각 — 1TB 재해시 비용 회피)"""
    try:
        st = path.stat()
        return {"path": rel, "size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        return {"path": rel, "size": -1, "mtime": 0}


def _save_failed_parts(
    case_id: str, parts: list[UnitPart], diagnostics: list[dict[str, Any]] | None = None
) -> None:
    """영구 임베딩 실패 파트를 data/failed_embeddings/{case_id}.jsonl에 보존 (재처리용)"""
    out_dir = settings.get_data_path("data/failed_embeddings")
    with (out_dir / f"{case_id}.jsonl").open("a", encoding="utf-8") as f:
        for i, p in enumerate(parts):
            rec = {
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "part": p.to_dict(),
                "diagnostics": (diagnostics[i] if diagnostics and i < len(diagnostics) else {}),
            }
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# 문서 파서가 지원하는 확장자
_DOCUMENT_EXTENSIONS = DocumentParser.SUPPORTED_TYPES


class IndexingPhase(StrEnum):
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
        embedding_service_factory: Any = None,
    ) -> None:
        """IndexingPipeline 초기화

        Args:
            case_store: 케이스 저장소 (기본: 새 CaseStore 생성)
            embedding_service_factory: model → EmbeddingService 팩토리 (테스트용 주입)
        """
        self.case_store = case_store or CaseStore()
        self._embedding_service_factory = embedding_service_factory

        # 진행률 추적 (case_id → IndexingProgress)
        self._progress: dict[str, IndexingProgress] = {}
        # 실행 중인 스레드 (case_id → Thread)
        self._threads: dict[str, threading.Thread] = {}
        # 취소 플래그 (case_id → Event)
        self._cancel_flags: dict[str, threading.Event] = {}

    def _create_embedding_service(self, model: str) -> Any:
        """서비스 프로필 임베딩 모델의 EmbeddingService"""
        if self._embedding_service_factory:
            return self._embedding_service_factory(model)
        from src.embeddings.embedding_service import EmbeddingService

        return EmbeddingService(model=model)

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

        삭제·재시작 등 즉시 후속 동작이 필요한 호출자가 사용. 취소된 실행은 색인을 쓰지 않는다.

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
        """동기 인덱싱 실행 (파싱·단위 생성과 임베딩을 동시에 돌리는 스트리밍 파이프라인)

        완료 시 케이스 통계: total_documents = 처리한 파일 수, total_chunks = 증거 단위 수.

        Args:
            case_id: 인덱싱할 케이스 ID
            cancel_flag: 취소 플래그 (set되면 중단, 색인은 쓰지 않음)

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
            files = case_source_files(case_meta)
            progress.total_files = len(files)
            logger.info(f"파일 수집 완료: {len(files)}개 ({case_id})")

            # 2~4. 파싱·단위 생성 → 임베딩 → 케이스 색인 저장 (파일이 없어도 빈 색인을 쓴다)
            progress.set_phase(IndexingPhase.PARSING)
            n_units, n_parts = self._run_pipeline(files, case_id, progress, cancel_flag)

            if cancel_flag and cancel_flag.is_set():
                progress.set_phase(IndexingPhase.CANCELLED)
                progress.completed_at = datetime.now()
                self.case_store.update_status(case_id, CaseStatus.CREATED)
                _save_indexing_log(progress, "cancelled")
                logger.info(f"인덱싱 취소됨: {case_id}")
                return progress

            progress.total_chunks = n_parts
            progress.stored_chunks = n_parts
            progress.set_phase(IndexingPhase.COMPLETED)
            progress.completed_at = datetime.now()

            # 케이스 통계 업데이트 + 상태 변경
            self.case_store.update_stats(
                case_id,
                total_documents=progress.processed_files,
                total_chunks=n_units,
            )
            self.case_store.update_status(case_id, CaseStatus.READY)

            logger.info(
                f"인덱싱 완료: {case_id} — 파일 {progress.processed_files}개, "
                f"단위 {n_units}개, 파트 {n_parts}개, 에러 {len(progress.errors)}개"
            )
            _save_indexing_log(progress, "completed")
            return progress

        except Exception as e:
            progress.set_phase(IndexingPhase.ERROR)
            progress.errors.append(str(e))
            progress.completed_at = datetime.now()
            with contextlib.suppress(Exception):  # 케이스 자체가 없을 수 있음
                self.case_store.set_error(case_id, str(e))

            # 에러 이력 저장
            _save_indexing_log(progress, "error")
            logger.error(f"인덱싱 파이프라인 에러: {case_id} — {e}")
            return progress

    # ------------------------------------------------------------------
    # 스트리밍 파이프라인:
    #   파싱·단위 생성(Process/Thread) → 메인(중복 제거) → part_queue(20) → 임베딩 워커
    #   → 끝나면 케이스 색인 1회 저장
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        files: list[tuple[Path, str]],
        case_id: str,
        progress: IndexingProgress,
        cancel_flag: threading.Event | None = None,
    ) -> tuple[int, int]:
        """파일 → 단위 → 임베딩 → 케이스 색인

        Args:
            files: [(파일 경로, 케이스 기준 상대 경로)]

        Returns:
            (색인에 들어간 단위 수, 파트 수). 취소되면 (0, 0)이고 색인은 그대로 둔다.
        """
        num_workers = _get_parsing_worker_count()
        embed_batch_size = settings.embed_batch_size
        embed_concurrent = max(1, settings.embed_max_concurrent)
        super_batch_size = embed_batch_size * embed_concurrent
        sentinel = None

        spec = profile_spec(settings.service_index_profile)
        embedder = self._create_embedding_service(spec.embed_model)
        # 증분: 같은 part_id·같은 입력의 기존 벡터 재사용
        reuse = case_index.reusable_vectors(case_id, spec.name)
        if reuse:
            logger.info(f"기존 색인 벡터 {len(reuse):,}개 재사용 가능 ({spec.name})")

        part_queue: queue.Queue[list[UnitPart] | None] = queue.Queue(maxsize=20)
        collector = UnitCollector()
        build_stats = BuildStats()
        vectors: dict[str, np.ndarray] = {}
        counter_lock = threading.Lock()
        state: dict[str, Any] = {"embedded": 0, "reused": 0, "failed": 0, "embed_errors": []}
        wall_times: dict[str, float] = {"parse": 0.0, "embed": 0.0}

        def embedding_worker() -> None:
            """part_queue → super-batch → (재사용 또는) 임베딩 → vectors"""
            buffer: list[UnitPart] = []
            active = 0.0
            batch_num = 0

            def flush(batch: list[UnitPart]) -> None:
                nonlocal active, batch_num
                if not batch:
                    return
                batch_num += 1
                t0 = time.monotonic()
                todo: list[tuple[UnitPart, str]] = []
                n_reused = 0
                for p in batch:
                    text = embed_input(p, spec.header)
                    v = reuse.get((p.part_id, input_hash(text)))
                    if v is None and not text.strip():
                        # 입력이 빈 파트(본문 없는 메일의 nohdr 입력 등) → 영벡터 (cos 0, 색인에는 남김)
                        v = np.zeros(embedder.get_dimension(), dtype=np.float16)
                    if v is not None:
                        vectors[p.part_id] = v
                        n_reused += 1
                    else:
                        todo.append((p, text))
                n_ok = 0
                if todo:
                    try:
                        embs = embedder.embed_texts([t for _, t in todo], cancel_event=cancel_flag)
                        failed = set(embedder._last_failed_indices)
                        for i, (p, _) in enumerate(todo):
                            if i not in failed and embs[i]:
                                vectors[p.part_id] = np.asarray(embs[i], dtype=np.float16)
                                n_ok += 1
                        if failed and not (cancel_flag and cancel_flag.is_set()):
                            idx_sorted = sorted(failed)
                            _save_failed_parts(
                                case_id,
                                [todo[i][0] for i in idx_sorted],
                                [embedder._last_failed_diagnostics.get(i, {}) for i in idx_sorted],
                            )
                            msg = (
                                f"임베딩 super-batch {batch_num} 부분 실패: "
                                f"{len(failed)}/{len(todo)}개 (failed_embeddings 저장)"
                            )
                            logger.warning(msg)
                            with counter_lock:
                                state["embed_errors"].append(msg)
                                state["failed"] += len(failed)
                    except Exception as e:
                        msg = f"임베딩 super-batch {batch_num} 실패 ({len(todo)}개): {e}"
                        try:
                            _save_failed_parts(case_id, [p for p, _ in todo])
                            msg += " → failed_embeddings 저장"
                        except Exception as save_e:
                            msg += f" + 저장 실패 (DATA LOSS): {save_e}"
                        logger.error(msg)
                        with counter_lock:
                            state["embed_errors"].append(msg)
                            state["failed"] += len(todo)
                active += time.monotonic() - t0
                with counter_lock:
                    state["embedded"] += n_ok + n_reused
                    state["reused"] += n_reused
                    progress.embedded_chunks = state["embedded"]
                    if n_ok or n_reused:
                        progress.last_success_at = time.time()

            while True:
                if cancel_flag and cancel_flag.is_set():
                    break
                try:
                    item = part_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if item is sentinel:
                    flush(buffer)
                    break
                buffer.extend(item)
                while len(buffer) >= super_batch_size and not (cancel_flag and cancel_flag.is_set()):
                    flush(buffer[:super_batch_size])
                    buffer = buffer[super_batch_size:]
            wall_times["embed"] = active
            progress.embed_wall_seconds = active

        embed_thread = threading.Thread(target=embedding_worker, daemon=True, name=f"embed-{case_id}")
        embed_thread.start()

        # 파일을 크기로 분류: 큰 파일은 ProcessPool(GIL 우회), 작은 파일은 ThreadPool.
        # 파일 수가 적으면 ProcessPool 부팅 비용이 더 커서 ThreadPool만 쓴다.
        sized: list[tuple[Path, str, int]] = []
        for f, rel in files:
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            sized.append((f, rel, size))
        sized.sort(key=lambda x: x[2], reverse=True)
        use_process = len(files) > num_workers * 2 and num_workers > 1
        large = [(f, r) for f, r, s in sized if use_process and s >= _PARSE_PROCESS_THRESHOLD_BYTES]
        small = [(f, r) for f, r, s in sized if not (use_process and s >= _PARSE_PROCESS_THRESHOLD_BYTES)]
        process_workers = min(num_workers, len(large)) if large else 0
        thread_workers = min(num_workers, len(small)) if small else 0
        logger.info(
            f"인덱싱 파이프라인 시작: 파일 {len(files)}개 (ProcessPool {len(large)} / ThreadPool {len(small)}), "
            f"프로필 {spec.name} ({spec.embed_model}), super-batch {super_batch_size}"
        )

        cancelled = False
        parse_start = time.monotonic()
        try:
            with ExitStack() as stack:
                pools: dict[str, Any] = {"proc": None, "thread": None, "restarts": 0}
                if process_workers > 0:
                    pools["proc"] = ProcessPoolExecutor(max_workers=process_workers, initializer=_subprocess_init)
                    stack.callback(lambda: pools["proc"] and pools["proc"].shutdown(wait=True, cancel_futures=True))
                if thread_workers > 0:
                    pools["thread"] = stack.enter_context(
                        ThreadPoolExecutor(max_workers=thread_workers, thread_name_prefix=f"parse-{case_id}")
                    )

                large_iter = iter(large)
                small_iter = iter(small)
                retry_large: list[tuple[Path, str]] = []  # 풀이 깨질 때 처리 중이던 큰 파일
                crash_count: dict[Path, int] = {}
                pending: dict[Future, tuple[Path, str, bool]] = {}  # future → (파일, 상대경로, 프로세스 풀 여부)
                max_pending = max(num_workers * 2, 4)

                def _thread_pool() -> ThreadPoolExecutor:
                    if pools["thread"] is None:
                        pools["thread"] = stack.enter_context(
                            ThreadPoolExecutor(max_workers=max(1, num_workers), thread_name_prefix=f"parse-{case_id}")
                        )
                    return pools["thread"]

                def _on_pool_broken() -> None:
                    """깨진 프로세스 풀 정리 → 처리 중이던 큰 파일 재시도 목록에 → 풀 재생성(상한까지)"""
                    broken = pools["proc"]
                    if broken is None:
                        return
                    for fut, (f, rel, in_proc) in list(pending.items()):
                        if in_proc and not (fut.done() and not fut.cancelled() and fut.exception() is None):
                            pending.pop(fut)
                            crash_count[f] = crash_count.get(f, 0) + 1
                            if crash_count[f] >= _MAX_CRASH_INVOLVEMENT:
                                msg = f"파싱 프로세스 비정상 종료에 {crash_count[f]}회 연루되어 건너뜀 ({f.name})"
                                with counter_lock:
                                    progress.errors.append(msg)
                                    progress.processed_files += 1
                                logger.warning(msg)
                            else:
                                retry_large.append((f, rel))
                    broken.shutdown(wait=False, cancel_futures=True)
                    if pools["restarts"] < _MAX_PROCESS_POOL_RESTARTS:
                        pools["restarts"] += 1
                        logger.warning(
                            f"파싱 프로세스 풀이 깨짐 → 다시 생성 ({pools['restarts']}/{_MAX_PROCESS_POOL_RESTARTS}), "
                            f"처리 중이던 파일 재시도"
                        )
                        pools["proc"] = ProcessPoolExecutor(max_workers=process_workers, initializer=_subprocess_init)
                    else:
                        logger.warning("파싱 프로세스 풀 재생성 한도 초과 → 남은 큰 파일은 ThreadPool로 처리")
                        pools["proc"] = None

                def _next_large() -> tuple[Path, str] | None:
                    """다음 큰 파일. 깨짐에 연루된 파일은 하나씩만(다른 프로세스 작업이 없을 때) 다시 보낸다
                    — 혼자 돌다 또 깨지면 그 파일이 원인으로 특정된다."""
                    if retry_large:
                        if any(in_proc for _, _, in_proc in pending.values()):
                            return None
                        return retry_large.pop(0)
                    return next(large_iter, None)

                def _submit_next() -> bool:
                    """대기 중인 파일 1개를 적절한 풀에 submit. 더 없으면 False."""
                    while True:
                        item = _next_large()
                        if item is None:
                            break
                        f, rel = item
                        if pools["proc"] is None:
                            if crash_count.get(f):
                                # 프로세스를 죽였을 수 있는 파일을 메인 프로세스 스레드에서 돌리지 않는다
                                msg = f"파싱 프로세스 비정상 종료에 연루되어 건너뜀 ({f.name})"
                                with counter_lock:
                                    progress.errors.append(msg)
                                    progress.processed_files += 1
                                logger.warning(msg)
                                continue
                            pending[_thread_pool().submit(_process_file_thread, f, rel, case_id)] = (f, rel, False)
                            return True
                        try:
                            fut = pools["proc"].submit(_process_file_subprocess, (str(f), rel, case_id))
                        except BrokenExecutor:
                            retry_large.insert(0, item)
                            _on_pool_broken()
                            continue
                        pending[fut] = (f, rel, True)
                        return True
                    for f, rel in small_iter:
                        pending[_thread_pool().submit(_process_file_thread, f, rel, case_id)] = (f, rel, False)
                        return True
                    return False

                for _ in range(max_pending):
                    if not _submit_next():
                        break

                while pending:
                    if cancel_flag and cancel_flag.is_set():
                        cancelled = True
                        for ex in (pools["proc"], pools["thread"]):
                            if ex is not None:
                                ex.shutdown(wait=False, cancel_futures=True)
                        break
                    done, _ = wait(list(pending), return_when=FIRST_COMPLETED, timeout=1.0)
                    for future in done:
                        if future not in pending:  # 풀 깨짐 처리에서 이미 재시도 목록으로 옮김
                            continue
                        file_path, _rel, in_proc = pending[future]
                        if in_proc and not future.cancelled() and isinstance(future.exception(), BrokenExecutor):
                            _on_pool_broken()
                            while len(pending) < max_pending and _submit_next():
                                pass
                            continue
                        pending.pop(future)
                        try:
                            units, stats, err = future.result()
                            build_stats.merge(BuildStats.from_dict(stats))
                            if err:
                                with counter_lock:
                                    progress.errors.append(err)
                                logger.warning(err)
                            elif units:
                                before = set(collector.units)
                                collector.add(units)
                                new_ids = [
                                    uid for uid in dict.fromkeys(u.unit_id for u in units)
                                    if uid not in before
                                ]
                                new_parts = [p for uid in new_ids for p in collector.units[uid].parts]
                                if new_parts:
                                    # 큐가 가득 차면 여기서 block → 자연스러운 backpressure
                                    part_queue.put(new_parts)
                                    with counter_lock:
                                        progress.parsed_chunks += len(new_parts)
                        except Exception as e:
                            msg = f"파싱 워커 실패 ({file_path.name}): {e}"
                            with counter_lock:
                                progress.errors.append(msg)
                            logger.warning(msg)
                        finally:
                            with counter_lock:
                                progress.processed_files += 1
                        while len(pending) < max_pending and _submit_next():
                            pass
        finally:
            wall_times["parse"] = time.monotonic() - parse_start
            progress.parse_wall_seconds = wall_times["parse"]
            part_queue.put(sentinel)
            embed_thread.join()

        progress.errors.extend(state["embed_errors"])
        if cancelled or (cancel_flag and cancel_flag.is_set()):
            return 0, 0

        # 결정적 행 순서: 파싱 완료 순서와 무관하게 (unit_id, 파트 번호)로 정렬
        parts = sorted(
            (p for p in collector.parts() if p.part_id in vectors),
            key=lambda p: (p.unit_id, int(p.part_id.rsplit("#", 1)[1])),
        )
        dim = int(next(iter(vectors.values())).shape[0]) if vectors else embedder.get_dimension()
        mat = (
            np.stack([vectors[p.part_id] for p in parts]).astype(np.float16)
            if parts else np.zeros((0, dim), dtype=np.float16)
        )
        progress.set_phase(IndexingPhase.STORING)
        t0 = time.monotonic()
        stats_out = build_stats.to_dict()
        stats_out["duplicates_removed"] = dict(collector.duplicates)
        stats_out["embedding"] = {
            "embedded": state["embedded"] - state["reused"],
            "reused": state["reused"],
            "failed": state["failed"],
        }
        case_index.write_index(
            case_id,
            parts,
            {spec.name: ProfileVectors(mat, spec.embed_model)},
            source_files=[_file_fingerprint(f, rel) for f, rel in files],
            build_stats=stats_out,
        )
        progress.store_wall_seconds = time.monotonic() - t0

        _check_failure_rate(
            progress=progress,
            stored=len(parts),
            store_errors=len(state["embed_errors"]),
            total_files=len(files),
        )
        n_units = len({p.unit_id for p in parts})
        logger.info(
            f"[BENCHMARK] 파싱·단위 {wall_times['parse']:.1f}s, 임베딩 {wall_times['embed']:.1f}s "
            f"(새로 {state['embedded'] - state['reused']}, 재사용 {state['reused']}, 실패 {state['failed']}), "
            f"색인 저장 {progress.store_wall_seconds:.1f}s — 단위 {n_units}, 파트 {len(parts)}, "
            f"중복 제거 {sum(collector.duplicates.values())}"
        )
        return n_units, len(parts)

    @staticmethod
    def _normalize_path(raw: str) -> Path:
        """윈도우 경로 문자열 정규화

        따옴표 제거, 양쪽 공백 제거, 경로 resolve.
        """
        cleaned = raw.strip().strip('"').strip("'").strip()
        return Path(cleaned)

    def _collect_files(self, case_meta: CaseMetadata) -> list[Path]:
        """케이스 데이터 소스에서 처리 대상 파일 수집 (collect_case_files 위임)"""
        return collect_case_files(case_meta)

    @staticmethod
    def _collect_files_impl(case_meta: CaseMetadata) -> list[Path]:
        """케이스 데이터 소스에서 처리 대상 파일 수집

        PST 경로: 파일이면 직접 추가, 폴더면 .pst/.ost 재귀 스캔
        문서 경로: 파일이면 직접 추가, 폴더면 지원 확장자 + .pst/.ost 재귀 스캔
        """
        files: list[Path] = []
        pst_extensions = {".pst", ".ost"}
        all_extensions = _DOCUMENT_EXTENSIONS | pst_extensions

        # PST 경로 처리
        for pst_path_str in case_meta.pst_paths:
            pst_path = IndexingPipeline._normalize_path(pst_path_str)
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
            doc_path = IndexingPipeline._normalize_path(doc_path_str)
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


# 인덱싱 실패 임계값 — 저장 시도 대비 실패 배치/파일 비율
_FAILURE_RATE_THRESHOLD = 0.10  # 10% 초과 시 abort


def collect_case_files(case_meta: CaseMetadata) -> list[Path]:
    """케이스 데이터 소스(PST 경로·문서 경로)에서 처리 대상 파일 목록 수집

    인덱싱 파이프라인과 scripts/unit_stats.py 가 같은 파일 목록을 쓰도록 공개한다.
    """
    return IndexingPipeline._collect_files_impl(case_meta)


class IndexingFailureRateExceeded(Exception):  # noqa: N818 — 공개 이름 유지
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
