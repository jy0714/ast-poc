"""인덱싱 파이프라인 유닛 테스트 (v4)

AST_FAKE_MODELS=1(가짜 임베더)과 임시 INDEX_DIR로 실제 경로 전체를 돈다:
파싱 → 증거 단위 → (중복 제거) → 임베딩 → 케이스 색인 저장.
"""

import tempfile
import threading
import time
from email.mime.text import MIMEText
from pathlib import Path

import pytest

from src.cases.case_store import CaseStatus, CaseStore
from src.index import case_index
from src.indexing import pipeline as pipeline_mod
from src.indexing.pipeline import (
    IndexingPhase,
    IndexingPipeline,
    IndexingProgress,
    case_source_files,
)
from src.units.schema import EvidenceUnit
from src.utils.config import settings

# === 테스트 헬퍼 ===


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture(autouse=True)
def _fake_models(tmpdir, monkeypatch):
    """가짜 임베더 + 임시 색인 폴더 + 실패 파트 기록 가로채기"""
    monkeypatch.setattr(settings, "ast_fake_models", True)
    monkeypatch.setattr(settings, "embed_dim", 32)
    monkeypatch.setattr(settings, "index_dir", str(tmpdir / "index"))
    monkeypatch.setattr(settings, "service_index_profile", "hdr-qwen3")
    monkeypatch.setattr(settings, "ollama_embed_model", "qwen3-embedding:4b-q8_0")
    saved: list = []
    monkeypatch.setattr(pipeline_mod, "_save_failed_parts", lambda cid, parts, diag=None: saved.extend(parts))
    case_index._cache.clear()
    yield saved
    case_index._cache.clear()


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
def pipeline(case_store):
    return IndexingPipeline(case_store=case_store)


def _eml(path: Path, subject: str, body: str, message_id: str) -> None:
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = message_id
    msg["Date"] = "Tue, 05 Mar 2024 14:22:00 +0900"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(msg.as_bytes())


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

    def test_progress_and_eta_use_bytes(self):
        """큰 파일부터 처리해도 진행률·ETA는 바이트 기준"""
        from datetime import datetime, timedelta

        p = IndexingProgress(case_id="t", phase=IndexingPhase.PARSING,
                             started_at=datetime.now() - timedelta(seconds=100))
        p.total_files, p.processed_files = 1000, 1
        p.total_bytes, p.processed_bytes = 1000, 500
        assert p.progress_percent == 50.0
        assert p.eta_seconds == pytest.approx(100, rel=0.05)

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




# === 파일 → 단위 ===


class TestBuildFile:
    def test_txt_to_units(self, tmpdir):
        """TXT 파일 → 문서 단위 (case_id·헤더·통계)"""
        doc = tmpdir / "test.txt"
        doc.write_text("이것은 테스트 문서입니다. 감사 보고서 내용입니다. " * 20, encoding="utf-8")
        units, stats, err = pipeline_mod._process_file_thread(doc, "test.txt", "c1")
        assert err is None and units
        assert all(isinstance(u, EvidenceUnit) for u in units)
        assert all(u.unit_type == "document" and u.meta["case_id"] == "c1" for u in units)
        assert all(u.parts and u.parts[0].header.startswith("[문서]") for u in units)
        assert stats["counts"]["units:document"] == len(units)

    def test_empty_txt(self, tmpdir):
        """빈 TXT → 단위 0개, 에러 없음"""
        doc = tmpdir / "empty.txt"
        doc.write_text("", encoding="utf-8")
        units, _, err = pipeline_mod._process_file_thread(doc, "empty.txt", "c1")
        assert units == [] and err is None

    def test_eml_is_email_unit(self, tmpdir):
        """문서 폴더의 EML도 이메일 단위"""
        _eml(tmpdir / "a.eml", "테스트 이메일", "EML 파싱 테스트 본문입니다.", "<m1@x>")
        units, _, err = pipeline_mod._process_file_thread(tmpdir / "a.eml", "a.eml", "c1")
        assert err is None
        assert units[0].unit_type == "email" and units[0].unit_id.startswith("em:")
        assert "테스트 이메일" in units[0].header

    def test_subprocess_worker_same_result(self, tmpdir):
        """ProcessPool 워커 함수도 같은 단위를 만든다 (같은 프로세스에서 직접 호출)"""
        _eml(tmpdir / "a.eml", "제목", "본문 내용입니다.", "<m2@x>")
        u1, _, _ = pipeline_mod._process_file_thread(tmpdir / "a.eml", "a.eml", "c1")
        u2, _, _ = pipeline_mod._process_file_subprocess((str(tmpdir / "a.eml"), "a.eml", "c1"))
        assert [u.unit_id for u in u1] == [u.unit_id for u in u2]

    def test_bad_file_reports_error(self, tmpdir):
        """파싱 실패 파일 → 에러 메시지 (예외 아님)"""
        bad = tmpdir / "bad.pdf"
        bad.write_bytes(b"not a real pdf")
        units, _, err = pipeline_mod._process_file_thread(bad, "bad.pdf", "c1")
        assert units == [] and err and "bad.pdf" in err

    def test_relative_paths_from_source_root(self, case_store, tmpdir):
        """상대 경로 = 소스 루트의 마지막 3개 요소 + 루트 기준 경로 (문서 ID의 근거)"""
        doc_dir = _create_test_files(tmpdir, {"a.txt": "x " * 10})
        (doc_dir / "sub").mkdir()
        (doc_dir / "sub" / "b.txt").write_text("y " * 10, encoding="utf-8")
        meta = case_store.create("경로", doc_paths=[str(doc_dir)])
        rels = sorted(rel for _, rel in case_source_files(meta))
        assert rels[0].endswith("/docs/a.txt") and rels[1].endswith("/docs/sub/b.txt")
        assert len(rels[0].split("/")) == 4


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


# === v4: 중복 제거·증분·실패 파트·취소 ===


class TestV4Pipeline:
    def test_index_written_and_searchable(self, pipeline, case_store, tmpdir):
        """완료 후 케이스 색인이 생기고 retrieve()로 검색된다"""
        from src.rag.retrieval import RetrievalConfig, retrieve

        doc_dir = _create_test_files(tmpdir, {
            "doc1.txt": "첫 번째 문서입니다. 감사 보고서 내용을 포함합니다. " * 10,
            "doc2.txt": "두 번째 문서입니다. 재무 분석 자료입니다. " * 10,
        })
        meta = case_store.create("검색", doc_paths=[str(doc_dir)])
        progress = pipeline.run(meta.case_id)
        idx = case_index.load(meta.case_id)
        assert idx.n_parts == progress.total_chunks
        assert idx.profiles() == ["hdr-qwen3"]
        assert case_store.get(meta.case_id).total_chunks == idx.n_units
        res = retrieve(meta.case_id, "재무 분석", RetrievalConfig(rerank=False))
        assert idx.part(res.hits[0].best_part_id).meta["source_file"].endswith("docs/doc2.txt")

    def test_empty_case_writes_empty_index(self, pipeline, case_store):
        meta = case_store.create("빈 케이스")
        pipeline.run(meta.case_id)
        assert case_index.load(meta.case_id).n_parts == 0

    def test_duplicate_email_across_custodians(self, pipeline, case_store, tmpdir):
        """같은 Message-ID 메일이 두 조사 대상자 폴더에 있으면 단위 1개 + folder_paths 병합"""
        root = tmpdir / "mail"
        _eml(root / "kim" / "Inbox" / "a.eml", "정산 건", "선급금 송금 관련 본문", "<dup@x>")
        _eml(root / "lee" / "Sent" / "a.eml", "정산 건", "선급금 송금 관련 본문", "<dup@x>")
        _eml(root / "lee" / "Sent" / "b.eml", "다른 메일", "점심 메뉴", "<other@x>")
        meta = case_store.create("중복", doc_paths=[str(root)])
        progress = pipeline.run(meta.case_id)
        assert progress.phase == IndexingPhase.COMPLETED
        idx = case_index.load(meta.case_id)
        assert idx.n_units == 2
        dup = [p for p in idx.parts if "dup@x" in p.meta.get("message_id", "")]
        assert len(dup) == 1
        folders = sorted(dup[0].meta["folder_paths"])
        assert len(folders) == 2
        assert folders[0].endswith("mail/kim/Inbox") and folders[1].endswith("mail/lee/Sent")
        assert idx.manifest["build_stats"]["duplicates_removed"] == {"email": 1}

    def test_incremental_reuses_vectors(self, case_store, tmpdir):
        """다시 돌리면 기존 파트 벡터를 재사용하고 새 파트만 임베딩"""
        from src.embeddings.embedding_service import EmbeddingService

        calls: list[int] = []

        class CountingService(EmbeddingService):
            def embed_texts(self, texts, cancel_event=None):
                calls.append(len(texts))
                return super().embed_texts(texts, cancel_event=cancel_event)

        pl = IndexingPipeline(
            case_store=case_store, embedding_service_factory=lambda m: CountingService(model=m)
        )
        doc_dir = _create_test_files(tmpdir, {"a.txt": "첫 문서 " * 20})
        meta = case_store.create("증분", doc_paths=[str(doc_dir)])
        pl.run(meta.case_id)
        first = sum(calls)
        assert first > 0
        (doc_dir / "b.txt").write_text("새 문서 내용 " * 20, encoding="utf-8")
        calls.clear()
        progress = pl.run(meta.case_id)
        assert progress.phase == IndexingPhase.COMPLETED
        idx = case_index.load(meta.case_id)
        assert idx.n_units == 2
        assert sum(calls) == 1  # b.txt 파트만 새로 임베딩
        assert idx.manifest["build_stats"]["embedding"]["reused"] == first

    def test_failed_parts_excluded_and_saved(self, case_store, tmpdir, _fake_models):
        """영구 임베딩 실패 파트는 색인(Dense·BM25)에서 빠지고 failed_embeddings에 저장"""
        from src.embeddings.embedding_service import EmbeddingService

        class FailSome(EmbeddingService):
            def embed_texts(self, texts, cancel_event=None):
                out = super().embed_texts(texts, cancel_event=cancel_event)
                bad = [i for i, t in enumerate(texts) if "실패" in t]
                for i in bad:
                    out[i] = []
                self._last_failed_indices = bad
                return out

        pl = IndexingPipeline(case_store=case_store, embedding_service_factory=lambda m: FailSome(model=m))
        doc_dir = _create_test_files(tmpdir, {"ok.txt": "정상 문서 " * 10, "bad.txt": "실패 문서 " * 10})
        meta = case_store.create("실패", doc_paths=[str(doc_dir)])
        pl.run(meta.case_id)
        idx = case_index.load(meta.case_id)
        assert idx.n_units == 1
        assert idx.bm25_scores("실패", header=False).sum() == 0
        assert _fake_models and "실패" in _fake_models[0].text

    def test_cancel_keeps_previous_index(self, pipeline, case_store, tmpdir):
        """취소된 실행은 색인을 쓰지 않는다"""
        doc_dir = _create_test_files(tmpdir, {"a.txt": "문서 " * 10})
        meta = case_store.create("취소", doc_paths=[str(doc_dir)])
        pipeline.run(meta.case_id)
        n_before = case_index.load(meta.case_id).n_parts
        (doc_dir / "b.txt").write_text("새 문서 " * 10, encoding="utf-8")
        flag = threading.Event()
        flag.set()
        progress = pipeline.run(meta.case_id, cancel_flag=flag)
        assert progress.phase == IndexingPhase.CANCELLED
        case_index._cache.clear()
        assert case_index.load(meta.case_id).n_parts == n_before

    def test_cancel_with_full_queue_does_not_hang(self, case_store, tmpdir):
        """임베딩 중 취소하면 파싱 쪽이 가득 찬 큐에서 멈추지 않고 끝난다"""
        from src.embeddings.embedding_service import EmbeddingService

        flag = threading.Event()

        class SlowCancel(EmbeddingService):
            def embed_texts(self, texts, cancel_event=None):
                flag.set()  # 첫 배치에서 취소 → 워커 종료, 큐는 곧 가득 참
                time.sleep(0.2)
                return super().embed_texts(texts, cancel_event=cancel_event)

        files = {f"f{i:03d}.txt": f"문서 {i} " * 5 for i in range(60)}
        doc_dir = _create_test_files(tmpdir, files)
        meta = case_store.create("취소 큐", doc_paths=[str(doc_dir)])
        pl = IndexingPipeline(case_store=case_store, embedding_service_factory=lambda m: SlowCancel(model=m))
        old = settings.embed_batch_size
        settings.embed_batch_size = 1
        try:
            done = threading.Event()
            result: list = []
            th = threading.Thread(target=lambda: (result.append(pl.run(meta.case_id, cancel_flag=flag)), done.set()))
            th.start()
            assert done.wait(60), "취소 후 파이프라인이 끝나지 않음"
            assert result[0].phase == IndexingPhase.CANCELLED
        finally:
            settings.embed_batch_size = old

    def test_checkpoint_reused_after_crash(self, case_store, tmpdir, monkeypatch):
        """색인 저장 전에 죽어도 임베딩은 체크포인트에 남고, 다시 돌리면 임베딩하지 않는다"""
        from src.embeddings.embedding_service import EmbeddingService
        from src.index.embed_cache import cache_dir

        calls: list[int] = []

        class CountingService(EmbeddingService):
            def embed_texts(self, texts, cancel_event=None):
                calls.append(len(texts))
                return super().embed_texts(texts, cancel_event=cancel_event)

        pl = IndexingPipeline(
            case_store=case_store, embedding_service_factory=lambda m: CountingService(model=m)
        )
        doc_dir = _create_test_files(tmpdir, {"a.txt": "첫 문서 " * 20, "b.txt": "둘째 문서 " * 20})
        meta = case_store.create("체크포인트", doc_paths=[str(doc_dir)])

        real_write = case_index.write_index

        def crash(*a, **k):
            raise RuntimeError("저장 직전 비정상 종료")

        monkeypatch.setattr(case_index, "write_index", crash)
        assert pl.run(meta.case_id).phase == IndexingPhase.ERROR
        first = sum(calls)
        assert first == 2
        assert list(cache_dir(meta.case_id, "hdr-qwen3").glob("shard_*.npz"))

        monkeypatch.setattr(case_index, "write_index", real_write)
        case_store.update_status(meta.case_id, CaseStatus.CREATED)
        calls.clear()
        progress = pl.run(meta.case_id)
        assert progress.phase == IndexingPhase.COMPLETED
        assert sum(calls) == 0
        idx = case_index.load(meta.case_id)
        assert idx.n_units == 2
        assert idx.manifest["build_stats"]["embedding"]["reused"] == first
        assert not cache_dir(meta.case_id, "hdr-qwen3").exists()  # 색인 저장 후 정리

    def test_ad_emails_dropped(self, pipeline, case_store, tmpdir):
        """제목이 (광고)·[광고]로 시작하는 메일은 색인에 넣지 않고 건수만 남긴다"""
        mail = tmpdir / "mail"
        _eml(mail / "a.eml", "(광고) 가을 할인 행사", "할인 안내 본문", "<ad1@x>")
        _eml(mail / "b.eml", "[광고]신제품 소식", "신제품 본문", "<ad2@x>")
        _eml(mail / "c.eml", "RE: (광고) 행사 참여 여부", "참여 검토 회신", "<re1@x>")
        _eml(mail / "d.eml", "3월 정산 건", "정산 내역 공유", "<ok1@x>")
        meta = case_store.create("광고", doc_paths=[str(mail)])
        progress = pipeline.run(meta.case_id)
        assert progress.phase == IndexingPhase.COMPLETED
        idx = case_index.load(meta.case_id)
        subjects = sorted(p.meta["subject"] for p in idx.parts)
        assert subjects == ["3월 정산 건", "RE: (광고) 행사 참여 여부"]
        assert idx.manifest["build_stats"]["counts"]["ads_removed"] == 2

    def test_progress_bytes(self, pipeline, case_store, tmpdir):
        """진행률은 바이트 기준으로 채워진다"""
        doc_dir = _create_test_files(tmpdir, {"a.txt": "문서 " * 50, "b.txt": "짧은 문서"})
        meta = case_store.create("바이트", doc_paths=[str(doc_dir)])
        progress = pipeline.run(meta.case_id)
        total = sum(f.stat().st_size for f in doc_dir.iterdir())
        assert progress.total_bytes == total
        assert progress.processed_bytes == total
        assert progress.progress_percent == 100.0
        d = progress.to_dict()
        assert d["total_bytes"] == total and d["processed_bytes"] == total


class TestProcessPoolCrash:
    def test_crashing_file_skipped_others_indexed(self, pipeline, case_store, tmpdir, monkeypatch):
        """파싱 프로세스가 죽어도 인덱싱 전체가 실패하지 않는다 — 원인 파일만 건너뛰고 나머지는 색인"""
        from tests.unit._parse_crash_helper import crash_or_parse

        monkeypatch.setattr(pipeline_mod, "_PARSE_PROCESS_THRESHOLD_BYTES", 0)
        monkeypatch.setattr(pipeline_mod, "_get_parsing_worker_count", lambda: 2)
        monkeypatch.setattr(pipeline_mod, "_process_file_subprocess", crash_or_parse)
        files = {f"doc{i}.txt": f"정상 문서 {i} 감사 자료 " * 20 for i in range(6)}
        files["crash.txt"] = "이 파일은 워커를 죽인다 " * 20
        doc_dir = _create_test_files(tmpdir, files)
        meta = case_store.create("크래시", doc_paths=[str(doc_dir)])

        progress = pipeline.run(meta.case_id)

        assert progress.phase.value == "completed"
        assert progress.processed_files == 7
        assert [e for e in progress.errors if "crash.txt" in e and "비정상 종료" in e]
        assert not [e for e in progress.errors if "doc" in e]
        sources = {p.meta["source_file"].rsplit("/", 1)[-1] for p in case_index.load(meta.case_id).parts}
        assert sources == {f"doc{i}.txt" for i in range(6)}
