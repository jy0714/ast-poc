"""인덱싱 파이프라인 유닛 테스트

Ollama 없이 테스트: VectorStoreService를 모킹하여
파싱 → 청킹 → 메타데이터 보강 단계까지 검증.
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cases.case_store import CaseStatus, CaseStore
from src.chunkers.chunker import Chunk
from src.indexing.pipeline import (
    IndexingPhase,
    IndexingPipeline,
    IndexingProgress,
)


# === 테스트 헬퍼 ===


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def case_store(tmpdir):
    db_path = tmpdir / "test.db"
    db_url = f"sqlite:///{db_path}"

    from src.db.database import reset_globals
    reset_globals()

    store = CaseStore(db_url=db_url)
    yield store
    reset_globals()


@pytest.fixture
def mock_vector_store():
    """add_chunks를 모킹한 VectorStore 팩토리"""
    def factory(case_id: str):
        store = MagicMock()
        store.add_chunks.side_effect = lambda chunks, rebuild_bm25=True: len(chunks)
        return store
    return factory


@pytest.fixture
def pipeline(case_store, mock_vector_store):
    return IndexingPipeline(
        case_store=case_store,
        vector_store_factory=mock_vector_store,
    )


def _create_test_files(tmpdir: Path, files: dict[str, str]) -> Path:
    """테스트 파일 생성, 파일들이 있는 디렉토리 반환"""
    doc_dir = tmpdir / "docs"
    doc_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (doc_dir / name).write_text(content, encoding="utf-8")
    return doc_dir


# === IndexingProgress 테스트 ===


class TestIndexingProgress:
    def test_progress_percent_zero(self):
        p = IndexingProgress(case_id="test")
        assert p.progress_percent == 0.0

    def test_progress_percent_partial(self):
        p = IndexingProgress(case_id="test", total_files=10, processed_files=3)
        assert p.progress_percent == 30.0

    def test_progress_percent_complete(self):
        p = IndexingProgress(case_id="test", total_files=5, processed_files=5)
        assert p.progress_percent == 100.0

    def test_is_running(self):
        p = IndexingProgress(case_id="test", phase=IndexingPhase.PARSING)
        assert p.is_running is True

    def test_is_not_running_idle(self):
        p = IndexingProgress(case_id="test", phase=IndexingPhase.IDLE)
        assert p.is_running is False

    def test_is_not_running_completed(self):
        p = IndexingProgress(case_id="test", phase=IndexingPhase.COMPLETED)
        assert p.is_running is False

    def test_to_dict(self):
        p = IndexingProgress(case_id="abc", total_files=10, processed_files=5)
        d = p.to_dict()
        assert d["case_id"] == "abc"
        assert d["progress_percent"] == 50.0
        assert d["total_files"] == 10


# === 파일 수집 테스트 ===


class TestCollectFiles:
    def test_collect_txt_files(self, pipeline, case_store, tmpdir):
        """문서 폴더에서 TXT 파일 수집"""
        doc_dir = _create_test_files(tmpdir, {
            "a.txt": "내용 A",
            "b.txt": "내용 B",
            "c.bmp": "이미지",  # 지원하지 않는 형식
        })
        meta = case_store.create("수집 테스트", doc_paths=[str(doc_dir)])
        files = pipeline._collect_files(meta)
        names = {f.name for f in files}
        assert "a.txt" in names
        assert "b.txt" in names
        assert "c.bmp" not in names

    def test_collect_single_file(self, pipeline, case_store, tmpdir):
        """개별 문서 파일 경로"""
        doc_file = tmpdir / "single.txt"
        doc_file.write_text("단일 파일", encoding="utf-8")
        meta = case_store.create("단일 파일", doc_paths=[str(doc_file)])
        files = pipeline._collect_files(meta)
        assert len(files) == 1
        assert files[0].name == "single.txt"

    def test_collect_missing_path_no_error(self, pipeline, case_store):
        """존재하지 않는 경로는 건너뛰기"""
        meta = case_store.create("없는 경로", doc_paths=["/nonexistent/path"])
        files = pipeline._collect_files(meta)
        assert files == []

    def test_collect_deduplicates(self, pipeline, case_store, tmpdir):
        """중복 경로 제거"""
        doc_dir = _create_test_files(tmpdir, {"dup.txt": "중복"})
        meta = case_store.create(
            "중복 제거",
            doc_paths=[str(doc_dir), str(doc_dir)],
        )
        files = pipeline._collect_files(meta)
        assert len(files) == 1

    def test_collect_nested_dirs(self, pipeline, case_store, tmpdir):
        """하위 디렉토리 재귀 스캔"""
        doc_dir = tmpdir / "docs"
        sub_dir = doc_dir / "sub"
        sub_dir.mkdir(parents=True)
        (doc_dir / "root.txt").write_text("루트", encoding="utf-8")
        (sub_dir / "nested.txt").write_text("중첩", encoding="utf-8")

        meta = case_store.create("재귀 스캔", doc_paths=[str(doc_dir)])
        files = pipeline._collect_files(meta)
        names = {f.name for f in files}
        assert "root.txt" in names
        assert "nested.txt" in names

    def test_collect_quoted_path(self, pipeline, case_store, tmpdir):
        """따옴표로 감싼 경로 정규화"""
        doc_dir = _create_test_files(tmpdir, {"q.txt": "따옴표 경로"})
        quoted = f'"{doc_dir}"'
        meta = case_store.create("따옴표", doc_paths=[quoted])
        files = pipeline._collect_files(meta)
        assert len(files) == 1
        assert files[0].name == "q.txt"

    def test_collect_pst_folder(self, pipeline, case_store, tmpdir):
        """PST 경로에 폴더를 넣으면 내부 .pst 파일 재귀 스캔"""
        pst_dir = tmpdir / "pst_data"
        pst_dir.mkdir()
        pst_file = pst_dir / "test.pst"
        pst_file.write_bytes(b"fake pst content")

        meta = case_store.create("PST 폴더", pst_paths=[str(pst_dir)])
        files = pipeline._collect_files(meta)
        assert len(files) == 1
        assert files[0].name == "test.pst"

    def test_collect_doc_folder_includes_pst(self, pipeline, case_store, tmpdir):
        """문서 폴더에 PST가 있으면 함께 수집"""
        mixed_dir = tmpdir / "mixed"
        mixed_dir.mkdir()
        (mixed_dir / "doc.txt").write_text("문서", encoding="utf-8")
        (mixed_dir / "mail.pst").write_bytes(b"fake pst")

        meta = case_store.create("혼합", doc_paths=[str(mixed_dir)])
        files = pipeline._collect_files(meta)
        names = {f.name for f in files}
        assert "doc.txt" in names
        assert "mail.pst" in names


# === 문서 처리 테스트 ===


class TestProcessDocument:
    def test_process_txt(self, pipeline, case_store, tmpdir):
        """TXT 파일 처리: 파싱 → 청킹 → 메타데이터"""
        doc_file = tmpdir / "test.txt"
        doc_file.write_text(
            "이것은 테스트 문서입니다. 충분히 긴 내용을 포함해야 합니다. " * 20,
            encoding="utf-8",
        )
        meta = case_store.create("문서 처리")
        progress = IndexingProgress(case_id=meta.case_id)
        chunks = pipeline._process_document(doc_file, meta.case_id, progress)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.metadata.get("case_id") == meta.case_id for c in chunks)
        assert all(c.metadata.get("topics") is not None for c in chunks)

    def test_process_empty_txt(self, pipeline, case_store, tmpdir):
        """빈 TXT 파일 → 청크 0개"""
        doc_file = tmpdir / "empty.txt"
        doc_file.write_text("", encoding="utf-8")
        meta = case_store.create("빈 문서")
        progress = IndexingProgress(case_id=meta.case_id)
        chunks = pipeline._process_document(doc_file, meta.case_id, progress)
        assert chunks == []

    def test_process_eml(self, pipeline, case_store, tmpdir):
        """EML 파일 처리"""
        from email.mime.text import MIMEText

        msg = MIMEText("EML 파싱 테스트 본문입니다. 인덱싱 파이프라인으로 처리합니다.", "plain", "utf-8")
        msg["Subject"] = "테스트 이메일"
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"

        eml_file = tmpdir / "test.eml"
        eml_file.write_bytes(msg.as_bytes())

        meta = case_store.create("EML 처리")
        progress = IndexingProgress(case_id=meta.case_id)
        chunks = pipeline._process_document(eml_file, meta.case_id, progress)

        assert len(chunks) > 0
        assert "테스트 이메일" in chunks[0].content or "EML 파싱" in chunks[0].content


# === 전체 파이프라인 실행 테스트 ===


class TestRun:
    def test_run_with_documents(self, pipeline, case_store, tmpdir):
        """문서 파일 인덱싱 전체 흐름"""
        doc_dir = _create_test_files(tmpdir, {
            "doc1.txt": "첫 번째 문서입니다. 감사 보고서 내용을 포함합니다. " * 10,
            "doc2.txt": "두 번째 문서입니다. 재무 분석 자료입니다. " * 10,
        })
        meta = case_store.create("전체 흐름", doc_paths=[str(doc_dir)])

        progress = pipeline.run(meta.case_id)

        assert progress.phase == IndexingPhase.COMPLETED
        assert progress.total_files == 2
        assert progress.processed_files == 2
        assert progress.total_chunks > 0
        assert len(progress.errors) == 0

        # 케이스 상태가 READY로 변경되었는지 확인
        updated_case = case_store.get(meta.case_id)
        assert updated_case.status == CaseStatus.READY
        assert updated_case.total_documents == 2
        assert updated_case.total_chunks > 0

    def test_run_empty_case(self, pipeline, case_store):
        """데이터 소스 없는 케이스 → 즉시 완료"""
        meta = case_store.create("빈 케이스")
        progress = pipeline.run(meta.case_id)

        assert progress.phase == IndexingPhase.COMPLETED
        assert progress.total_files == 0
        assert progress.total_chunks == 0

    def test_run_with_errors_continues(self, pipeline, case_store, tmpdir):
        """일부 파일 에러 시 나머지 파일은 계속 처리"""
        doc_dir = tmpdir / "docs"
        doc_dir.mkdir()
        # 정상 파일
        (doc_dir / "good.txt").write_text("정상 문서 내용입니다. " * 10, encoding="utf-8")
        # 손상된 PDF (파싱 실패할 파일)
        (doc_dir / "bad.pdf").write_bytes(b"not a real pdf")

        meta = case_store.create("에러 포함", doc_paths=[str(doc_dir)])
        progress = pipeline.run(meta.case_id)

        assert progress.phase == IndexingPhase.COMPLETED
        assert progress.processed_files == 2
        assert len(progress.errors) >= 1  # bad.pdf 에러
        assert progress.total_chunks > 0  # good.txt 청크는 저장됨

    def test_run_case_not_found(self, pipeline):
        """존재하지 않는 케이스 → 에러"""
        progress = pipeline.run("nonexistent")
        assert progress.phase == IndexingPhase.ERROR
        assert len(progress.errors) > 0

    def test_run_updates_case_status(self, pipeline, case_store, tmpdir):
        """인덱싱 중 케이스 상태 전이 확인"""
        doc_dir = _create_test_files(tmpdir, {"test.txt": "테스트 " * 10})
        meta = case_store.create("상태 확인", doc_paths=[str(doc_dir)])

        assert case_store.get(meta.case_id).status == CaseStatus.CREATED
        pipeline.run(meta.case_id)
        assert case_store.get(meta.case_id).status == CaseStatus.READY


# === 취소 테스트 ===


class TestCancel:
    def test_cancel_not_running(self, pipeline):
        """실행 중이 아닌 케이스 취소 → False"""
        assert pipeline.cancel("abc") is False

    def test_cancel_running(self, pipeline, case_store, tmpdir):
        """실행 중인 인덱싱 취소"""
        # 많은 파일로 시간 확보
        doc_dir = tmpdir / "docs"
        doc_dir.mkdir()
        for i in range(20):
            (doc_dir / f"file_{i}.txt").write_text(f"내용 {i} " * 50, encoding="utf-8")

        meta = case_store.create("취소 테스트", doc_paths=[str(doc_dir)])
        pipeline.run_async(meta.case_id)

        # 잠시 대기 후 취소
        time.sleep(0.2)
        if pipeline.is_running(meta.case_id):
            result = pipeline.cancel(meta.case_id)
            assert result is True

            # 스레드 완료 대기 (DB 커넥션 해제)
            thread = pipeline._threads.get(meta.case_id)
            if thread is not None:
                thread.join(timeout=10)


# === 비동기 실행 테스트 ===


class TestRunAsync:
    def test_run_async_starts(self, pipeline, case_store, tmpdir):
        """비동기 실행 시작 및 완료"""
        doc_dir = _create_test_files(tmpdir, {"async.txt": "비동기 테스트 " * 10})
        meta = case_store.create("비동기", doc_paths=[str(doc_dir)])

        pipeline.run_async(meta.case_id)

        # 스레드 완료 대기
        thread = pipeline._threads.get(meta.case_id)
        if thread is not None:
            thread.join(timeout=10)

        final = pipeline.get_progress(meta.case_id)
        assert final.phase == IndexingPhase.COMPLETED

    def test_run_async_already_running(self, pipeline, case_store, tmpdir):
        """이미 실행 중이면 기존 progress 반환"""
        doc_dir = tmpdir / "docs"
        doc_dir.mkdir()
        for i in range(10):
            (doc_dir / f"f{i}.txt").write_text(f"내용 {i} " * 50, encoding="utf-8")

        meta = case_store.create("중복 실행", doc_paths=[str(doc_dir)])
        pipeline.run_async(meta.case_id)
        time.sleep(0.1)

        # 두 번째 호출 — 새 스레드 생성 안 됨
        p2 = pipeline.run_async(meta.case_id)
        assert p2.case_id == meta.case_id

        # 스레드 완료 대기 (DB 커넥션 해제를 위해 join)
        thread = pipeline._threads.get(meta.case_id)
        if thread is not None:
            thread.join(timeout=10)


# === get_progress 테스트 ===


class TestGetProgress:
    def test_unknown_case(self, pipeline):
        """알 수 없는 케이스 → idle 상태 반환"""
        p = pipeline.get_progress("unknown")
        assert p.phase == IndexingPhase.IDLE
        assert p.case_id == "unknown"


# === _save_indexing_log 안전망 (P1-B) ===


class TestSaveIndexingLog:
    """_save_indexing_log가 케이스 부재/DB 문제에서 graceful degrade"""

    def test_skip_when_case_missing(self, case_store, caplog):
        """케이스가 cases 테이블에 없으면 INSERT skip + INFO 로그"""
        import logging

        from src.indexing.pipeline import IndexingProgress, _save_indexing_log

        progress = IndexingProgress(case_id="never_existed_xyz")
        progress.total_files = 5
        progress.processed_files = 5

        with caplog.at_level(logging.INFO):
            # 예외 없이 통과해야 함
            _save_indexing_log(progress, "completed")

        # FK violation ERROR가 아니라 INFO로 처리됨
        msgs = [r.message for r in caplog.records]
        assert any("이미 삭제됨" in m for m in msgs)
        assert not any("FOREIGN KEY constraint" in m for m in msgs)


# === stalled / 운영 가시성 필드 (P3-E) ===


class TestProgressStalledField:
    """to_dict()가 last_success_at, stalled, encrypted_pdfs, scan_pdfs_no_ocr 노출"""

    def test_to_dict_has_new_fields(self):
        from src.indexing.pipeline import IndexingProgress

        p = IndexingProgress(case_id="x")
        d = p.to_dict()
        assert "last_success_at" in d
        assert "stalled" in d
        assert "encrypted_pdfs" in d
        assert "scan_pdfs_no_ocr" in d
        assert d["last_success_at"] is None
        assert d["stalled"] is False
        assert d["encrypted_pdfs"] == []
        assert d["scan_pdfs_no_ocr"] == []
