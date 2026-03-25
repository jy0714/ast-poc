"""인덱싱 API 엔드포인트 테스트"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import indexing as indexing_module
from src.cases.case_store import CaseStatus, CaseStore
from src.indexing.pipeline import IndexingPipeline


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def setup(tmpdir):
    """테스트용 CaseStore + Pipeline 교체 (SQLite)"""
    db_path = tmpdir / "test.db"
    db_url = f"sqlite:///{db_path}"

    from src.db.database import reset_globals
    reset_globals()

    store = CaseStore(db_url=db_url)

    def mock_vs_factory(case_id):
        vs = MagicMock()
        vs.add_chunks.side_effect = lambda chunks: len(chunks)
        return vs

    pipeline = IndexingPipeline(
        case_store=store,
        vector_store_factory=mock_vs_factory,
    )

    # 모듈 전역 교체
    orig_store = indexing_module._case_store
    orig_pipeline = indexing_module._pipeline
    indexing_module._case_store = store
    indexing_module._pipeline = pipeline

    yield store, pipeline

    indexing_module._case_store = orig_store
    indexing_module._pipeline = orig_pipeline
    reset_globals()


@pytest.fixture
def client(setup):
    return TestClient(app)


@pytest.fixture
def store(setup):
    return setup[0]


@pytest.fixture
def pipeline(setup):
    return setup[1]


# === 인덱싱 시작 API ===


class TestStartAPI:
    def test_start_indexing(self, client, store, tmpdir):
        """POST /start — 인덱싱 시작"""
        doc_dir = tmpdir / "docs"
        doc_dir.mkdir()
        (doc_dir / "test.txt").write_text("테스트 문서입니다. " * 10, encoding="utf-8")
        meta = store.create("시작 테스트", doc_paths=[str(doc_dir)])

        resp = client.post("/api/admin/indexing/start", json={"case_id": meta.case_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["case_id"] == meta.case_id

        # 완료 대기
        for _ in range(50):
            r = client.get(f"/api/admin/indexing/progress/{meta.case_id}")
            if r.json()["status"] in ("completed", "error"):
                break
            time.sleep(0.1)

    def test_start_not_found(self, client):
        """POST /start — 404"""
        resp = client.post("/api/admin/indexing/start", json={"case_id": "fake"})
        assert resp.status_code == 404

    def test_start_archived_case(self, client, store):
        """POST /start — 보관된 케이스 409"""
        meta = store.create("보관 케이스")
        store.update_status(meta.case_id, CaseStatus.ARCHIVED)
        resp = client.post("/api/admin/indexing/start", json={"case_id": meta.case_id})
        assert resp.status_code == 409


# === 진행률 조회 API ===


class TestProgressAPI:
    def test_progress_idle(self, client, store):
        """GET /progress/{case_id} — idle 상태"""
        meta = store.create("진행률 테스트")
        resp = client.get(f"/api/admin/indexing/progress/{meta.case_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"
        assert data["progress_percent"] == 0.0

    def test_progress_not_found(self, client):
        """GET /progress/{case_id} — 404"""
        resp = client.get("/api/admin/indexing/progress/fake_id")
        assert resp.status_code == 404


# === 중단 API ===


class TestStopAPI:
    def test_stop_not_running(self, client, store):
        """POST /stop/{case_id} — 실행 중이 아닌 경우 409"""
        meta = store.create("중단 테스트")
        resp = client.post(f"/api/admin/indexing/stop/{meta.case_id}")
        assert resp.status_code == 409


# === 증분 인덱싱 API ===


class TestIncrementAPI:
    def test_increment_not_found(self, client):
        """POST /increment/{case_id} — 404"""
        resp = client.post("/api/admin/indexing/increment/fake", json={
            "pst_paths": [],
            "doc_paths": ["/some/path"],
        })
        assert resp.status_code == 404

    def test_increment_no_sources(self, client, store):
        """POST /increment/{case_id} — 소스 없으면 400"""
        meta = store.create("증분 테스트")
        resp = client.post(f"/api/admin/indexing/increment/{meta.case_id}", json={
            "pst_paths": [],
            "doc_paths": [],
        })
        assert resp.status_code == 400

    def test_increment_with_docs(self, client, store, tmpdir):
        """POST /increment/{case_id} — 증분 인덱싱"""
        doc_dir = tmpdir / "docs"
        doc_dir.mkdir()
        (doc_dir / "inc.txt").write_text("증분 데이터입니다. " * 10, encoding="utf-8")

        meta = store.create("증분 실행")
        resp = client.post(f"/api/admin/indexing/increment/{meta.case_id}", json={
            "doc_paths": [str(doc_dir)],
        })
        assert resp.status_code == 200
        assert resp.json()["case_id"] == meta.case_id

        # 완료 대기
        for _ in range(50):
            r = client.get(f"/api/admin/indexing/progress/{meta.case_id}")
            if r.json()["status"] in ("completed", "error"):
                break
            time.sleep(0.1)
