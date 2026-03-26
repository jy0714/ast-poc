"""대시보드 API 단위 테스트"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes.dashboard import aggregate_dashboard
from src.cases.case_store import CaseStatus, CaseStore


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
def client(case_store):
    from src.api.routes import dashboard
    dashboard._case_store = case_store
    return TestClient(app)


# === aggregate_dashboard 단위 테스트 ===


class TestAggregateDashboard:
    """순수 집계 함수 테스트"""

    def test_empty_metadatas(self):
        """메타데이터 없으면 빈 결과"""
        result = aggregate_dashboard("case1", [])
        assert result["source_type_counts"] == {}
        assert result["top_participants"] == []
        assert result["participant_network"].nodes == []
        assert result["participant_network"].edges == []
        assert result["timeline"] == []
        assert result["top_topics"] == []

    def test_source_type_counts(self):
        """소스 타입 카운트"""
        metas = [
            {"source_type": "email"},
            {"source_type": "email"},
            {"source_type": "teams_chat"},
            {"source_type": "document"},
        ]
        result = aggregate_dashboard("case1", metas)
        assert result["source_type_counts"]["email"] == 2
        assert result["source_type_counts"]["teams_chat"] == 1
        assert result["source_type_counts"]["document"] == 1

    def test_participant_counts(self):
        """참여자 빈도 집계"""
        metas = [
            {"source_type": "email", "participants": json.dumps(["Alice", "Bob"])},
            {"source_type": "email", "participants": json.dumps(["Alice", "Charlie"])},
            {"source_type": "teams_chat", "participants": json.dumps(["Alice"])},
        ]
        result = aggregate_dashboard("case1", metas)
        ids = {p.id: p.message_count for p in result["top_participants"]}
        assert ids["Alice"] == 3
        assert ids["Bob"] == 1
        assert ids["Charlie"] == 1

    def test_network_edges(self):
        """동일 청크 참여자 간 엣지 생성"""
        metas = [
            {"source_type": "email", "participants": json.dumps(["Alice", "Bob"])},
            {"source_type": "email", "participants": json.dumps(["Alice", "Bob"])},
            {"source_type": "email", "participants": json.dumps(["Alice", "Charlie"])},
            {"source_type": "email", "participants": json.dumps(["Alice", "Charlie"])},
        ]
        result = aggregate_dashboard("case1", metas)
        edges = {(e.source, e.target): e.weight for e in result["participant_network"].edges}
        assert edges[("Alice", "Bob")] == 2
        assert edges[("Alice", "Charlie")] == 2

    def test_edge_minimum_weight(self):
        """weight < 2인 엣지는 제외"""
        metas = [
            {"source_type": "email", "participants": json.dumps(["Alice", "Bob"])},
            {"source_type": "email", "participants": json.dumps(["Alice", "Charlie"])},
            {"source_type": "email", "participants": json.dumps(["Alice", "Charlie"])},
        ]
        result = aggregate_dashboard("case1", metas)
        edges = result["participant_network"].edges
        # Alice-Bob은 1회라 제외, Alice-Charlie는 2회라 포함
        edge_pairs = [(e.source, e.target) for e in edges]
        assert ("Alice", "Charlie") in edge_pairs
        assert ("Alice", "Bob") not in edge_pairs

    def test_timeline_monthly_bucketing(self):
        """월별 타임라인 집계"""
        metas = [
            {"source_type": "email", "date_range_start": "2025-01-15T10:00:00"},
            {"source_type": "email", "date_range_start": "2025-01-20T14:00:00"},
            {"source_type": "teams_chat", "date_range_start": "2025-02-10T09:00:00"},
        ]
        result = aggregate_dashboard("case1", metas)
        timeline = {t.month: t for t in result["timeline"]}
        assert timeline["2025-01"].email_count == 2
        assert timeline["2025-02"].teams_chat_count == 1

    def test_timeline_sorted(self):
        """타임라인이 월별로 정렬되어야 함"""
        metas = [
            {"source_type": "email", "date_range_start": "2025-03-01"},
            {"source_type": "email", "date_range_start": "2025-01-01"},
        ]
        result = aggregate_dashboard("case1", metas)
        months = [t.month for t in result["timeline"]]
        assert months == ["2025-01", "2025-03"]

    def test_topics_aggregation(self):
        """토픽 빈도 집계"""
        metas = [
            {"source_type": "email", "topics": json.dumps(["감사", "계약"])},
            {"source_type": "email", "topics": json.dumps(["감사", "보고서"])},
            {"source_type": "document", "topics": json.dumps(["감사"])},
        ]
        result = aggregate_dashboard("case1", metas)
        topics = {t.topic: t.count for t in result["top_topics"]}
        assert topics["감사"] == 3
        assert topics["계약"] == 1
        assert topics["보고서"] == 1

    def test_participant_source_types(self):
        """참여자별 소스 타입 목록"""
        metas = [
            {"source_type": "email", "participants": json.dumps(["Alice"])},
            {"source_type": "teams_chat", "participants": json.dumps(["Alice"])},
        ]
        result = aggregate_dashboard("case1", metas)
        alice = next(p for p in result["top_participants"] if p.id == "Alice")
        assert "email" in alice.source_types
        assert "teams_chat" in alice.source_types

    def test_participants_already_list(self):
        """participants가 이미 list인 경우 (deserialize 후)"""
        metas = [
            {"source_type": "email", "participants": ["Alice", "Bob"]},
        ]
        result = aggregate_dashboard("case1", metas)
        ids = {p.id for p in result["top_participants"]}
        assert "Alice" in ids
        assert "Bob" in ids


# === API 엔드포인트 테스트 ===


class TestDashboardAPI:
    """HTTP 엔드포인트 테스트"""

    def test_case_not_found(self, client):
        """존재하지 않는 케이스 → 404"""
        resp = client.get("/api/analyst/dashboard/nonexistent")
        assert resp.status_code == 404

    def test_case_not_ready(self, client, case_store):
        """인덱싱 미완료 케이스 → 409"""
        meta = case_store.create("테스트")
        resp = client.get(f"/api/analyst/dashboard/{meta.case_id}")
        assert resp.status_code == 409

    @patch("src.api.routes.dashboard.VectorStoreService")
    def test_dashboard_success(self, mock_vs_cls, client, case_store):
        """정상 조회"""
        meta = case_store.create("대시보드 테스트")
        case_store.update_status(meta.case_id, CaseStatus.INDEXING)
        case_store.update_status(meta.case_id, CaseStatus.READY)

        # ChromaDB mock
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.get.return_value = {
            "metadatas": [
                {"source_type": "email", "participants": json.dumps(["Alice", "Bob"]),
                 "date_range_start": "2025-01-15", "topics": json.dumps(["감사"])},
                {"source_type": "email", "participants": json.dumps(["Alice"]),
                 "date_range_start": "2025-02-10", "topics": json.dumps(["계약"])},
                {"source_type": "teams_chat", "participants": json.dumps(["Bob", "Charlie"]),
                 "date_range_start": "2025-02-20", "topics": json.dumps(["회의"])},
            ]
        }
        mock_vs = MagicMock()
        mock_vs._get_collection.return_value = mock_collection
        mock_vs_cls.return_value = mock_vs

        resp = client.get(f"/api/analyst/dashboard/{meta.case_id}")
        assert resp.status_code == 200

        data = resp.json()
        assert data["case_id"] == meta.case_id
        assert data["case_name"] == "대시보드 테스트"
        assert data["total_chunks"] == 3
        assert data["source_type_counts"]["email"] == 2
        assert data["source_type_counts"]["teams_chat"] == 1
        assert len(data["timeline"]) == 2  # 2025-01, 2025-02
        assert len(data["top_topics"]) == 3

    @patch("src.api.routes.dashboard.VectorStoreService")
    def test_dashboard_empty_collection(self, mock_vs_cls, client, case_store):
        """빈 컬렉션"""
        meta = case_store.create("빈 케이스")
        case_store.update_status(meta.case_id, CaseStatus.INDEXING)
        case_store.update_status(meta.case_id, CaseStatus.READY)

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_vs = MagicMock()
        mock_vs._get_collection.return_value = mock_collection
        mock_vs_cls.return_value = mock_vs

        resp = client.get(f"/api/analyst/dashboard/{meta.case_id}")
        assert resp.status_code == 200

        data = resp.json()
        assert data["total_chunks"] == 0
        assert data["source_type_counts"] == {}
        assert data["top_participants"] == []

    @patch("src.api.routes.dashboard.VectorStoreService")
    def test_dashboard_archived_case(self, mock_vs_cls, client, case_store):
        """보관된 케이스도 조회 가능"""
        meta = case_store.create("보관 케이스")
        case_store.update_status(meta.case_id, CaseStatus.INDEXING)
        case_store.update_status(meta.case_id, CaseStatus.READY)
        case_store.update_status(meta.case_id, CaseStatus.ARCHIVED)

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_vs = MagicMock()
        mock_vs._get_collection.return_value = mock_collection
        mock_vs_cls.return_value = mock_vs

        resp = client.get(f"/api/analyst/dashboard/{meta.case_id}")
        assert resp.status_code == 200
