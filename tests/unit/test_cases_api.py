"""케이스 관리 API 엔드포인트 테스트"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import cases as cases_module
from src.cases.case_store import CaseStatus, CaseStore


@pytest.fixture
def store():
    """임시 SQLite DB 기반 CaseStore로 교체"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db_url = f"sqlite:///{db_path}"

        from src.db.database import reset_globals
        reset_globals()

        temp_store = CaseStore(db_url=db_url)
        original = cases_module._store
        cases_module._store = temp_store
        yield temp_store
        cases_module._store = original
        reset_globals()


@pytest.fixture
def client(store):
    """TestClient (store fixture에 의존)"""
    return TestClient(app)


# === 생성 API ===


class TestCreateAPI:
    def test_create_case(self, client: TestClient):
        """POST / — 케이스 생성"""
        resp = client.post("/api/admin/cases/", json={
            "name": "API 테스트 케이스",
            "description": "테스트용",
            "pst_paths": ["/data/test.pst"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "API 테스트 케이스"
        assert data["status"] == "created"
        assert len(data["case_id"]) == 12
        assert "/data/test.pst" in data["pst_paths"]

    def test_create_case_empty_name(self, client: TestClient):
        """POST / — 빈 이름 거부"""
        resp = client.post("/api/admin/cases/", json={"name": "   "})
        assert resp.status_code == 400

    def test_create_case_minimal(self, client: TestClient):
        """POST / — 이름만으로 생성"""
        resp = client.post("/api/admin/cases/", json={"name": "최소 케이스"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["description"] == ""
        assert data["pst_paths"] == []


# === 목록 조회 API ===


class TestListAPI:
    def test_list_empty(self, client: TestClient):
        """GET / — 빈 목록"""
        resp = client.get("/api/admin/cases/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_multiple(self, client: TestClient):
        """GET / — 여러 케이스"""
        client.post("/api/admin/cases/", json={"name": "케이스 1"})
        client.post("/api/admin/cases/", json={"name": "케이스 2"})
        resp = client.get("/api/admin/cases/")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_list_filter_by_status(self, client: TestClient, store: CaseStore):
        """GET /?status=created — 상태 필터"""
        r1 = client.post("/api/admin/cases/", json={"name": "A"})
        r2 = client.post("/api/admin/cases/", json={"name": "B"})
        # B를 archived로 변경
        case_b_id = r2.json()["case_id"]
        store.update_status(case_b_id, CaseStatus.ARCHIVED)

        resp = client.get("/api/admin/cases/?status=created")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["name"] == "A"

    def test_list_filter_invalid_status(self, client: TestClient):
        """GET /?status=invalid — 잘못된 상태 필터"""
        resp = client.get("/api/admin/cases/?status=invalid")
        assert resp.status_code == 400


# === 상세 조회 API ===


class TestGetAPI:
    def test_get_case(self, client: TestClient):
        """GET /{case_id} — 상세 조회"""
        create_resp = client.post("/api/admin/cases/", json={"name": "조회 테스트"})
        case_id = create_resp.json()["case_id"]

        resp = client.get(f"/api/admin/cases/{case_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "조회 테스트"

    def test_get_not_found(self, client: TestClient):
        """GET /{case_id} — 404"""
        resp = client.get("/api/admin/cases/nonexistent")
        assert resp.status_code == 404


# === 데이터 소스 업데이트 API ===


class TestUpdateSourcesAPI:
    def test_add_sources(self, client: TestClient):
        """PATCH /{case_id} — 데이터 소스 추가"""
        create_resp = client.post("/api/admin/cases/", json={"name": "소스 추가"})
        case_id = create_resp.json()["case_id"]

        resp = client.patch(f"/api/admin/cases/{case_id}", json={
            "pst_paths": ["/new.pst"],
            "doc_paths": ["/new/docs"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "/new.pst" in data["pst_paths"]
        assert "/new/docs" in data["doc_paths"]

    def test_add_sources_not_found(self, client: TestClient):
        """PATCH /{case_id} — 404"""
        resp = client.patch("/api/admin/cases/fake", json={"pst_paths": ["/x.pst"]})
        assert resp.status_code == 404


# === 삭제 API ===


class TestDeleteAPI:
    def test_delete_case(self, client: TestClient):
        """DELETE /{case_id} — 삭제"""
        create_resp = client.post("/api/admin/cases/", json={"name": "삭제 대상"})
        case_id = create_resp.json()["case_id"]

        resp = client.delete(f"/api/admin/cases/{case_id}")
        assert resp.status_code == 200
        assert resp.json()["message"] == "케이스가 삭제되었습니다"

        # 삭제 후 조회 시 404
        resp = client.get(f"/api/admin/cases/{case_id}")
        assert resp.status_code == 404

    def test_delete_not_found(self, client: TestClient):
        """DELETE /{case_id} — 404"""
        resp = client.delete("/api/admin/cases/nonexistent")
        assert resp.status_code == 404


# === 보관 API ===


class TestArchiveAPI:
    def test_archive_case(self, client: TestClient, store: CaseStore):
        """POST /{case_id}/archive — 보관"""
        create_resp = client.post("/api/admin/cases/", json={"name": "보관 대상"})
        case_id = create_resp.json()["case_id"]

        # created → archived 는 유효한 전이
        resp = client.post(f"/api/admin/cases/{case_id}/archive")
        assert resp.status_code == 200
        assert resp.json()["status"] == "archived"

    def test_archive_already_archived(self, client: TestClient, store: CaseStore):
        """POST /{case_id}/archive — 이미 보관 상태면 409"""
        create_resp = client.post("/api/admin/cases/", json={"name": "이미 보관"})
        case_id = create_resp.json()["case_id"]
        store.update_status(case_id, CaseStatus.ARCHIVED)

        resp = client.post(f"/api/admin/cases/{case_id}/archive")
        assert resp.status_code == 409

    def test_archive_not_found(self, client: TestClient):
        """POST /{case_id}/archive — 404"""
        resp = client.post("/api/admin/cases/nonexistent/archive")
        assert resp.status_code == 404
