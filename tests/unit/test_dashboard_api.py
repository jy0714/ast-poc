"""대시보드 API 단위 테스트"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes.dashboard import aggregate_dashboard
from src.cases.case_store import CaseStatus, CaseStore
from src.index import case_index
from src.index.case_index import ProfileVectors
from src.units.schema import UnitPart
from src.utils.config import settings


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


def _write_units(case_id: str, metas: list[tuple[str, str, dict]]) -> None:
    """(unit_id, unit_type, meta) 목록으로 케이스 색인 작성 (단위당 파트 1개, 첫 단위는 파트 2개)"""
    parts = []
    for i, (uid, utype, meta) in enumerate(metas):
        n = 2 if i == 0 else 1
        for k in range(1, n + 1):
            parts.append(UnitPart(f"{uid}#{k}", uid, utype, "[헤더]", f"본문 {k}", 2, meta))
    vecs = np.zeros((len(parts), 4), dtype=np.float32)
    case_index.write_index(
        case_id, parts, {"hdr-qwen3": ProfileVectors(vecs, settings.ollama_embed_model)}
    )


def _ready(case_store, name: str) -> str:
    meta = case_store.create(name)
    case_store.update_status(meta.case_id, CaseStatus.INDEXING)
    case_store.update_status(meta.case_id, CaseStatus.READY)
    return meta.case_id


class TestDashboardAPI:
    """HTTP 엔드포인트 테스트"""

    @pytest.fixture(autouse=True)
    def _tmp_index(self, tmpdir, monkeypatch):
        monkeypatch.setattr(settings, "index_dir", str(tmpdir / "index"))
        case_index._cache.clear()
        yield
        case_index._cache.clear()

    def test_case_not_found(self, client):
        """존재하지 않는 케이스 → 404"""
        resp = client.get("/api/analyst/dashboard/nonexistent")
        assert resp.status_code == 404

    def test_case_not_ready(self, client, case_store):
        """인덱싱 미완료 케이스 → 409"""
        meta = case_store.create("테스트")
        resp = client.get(f"/api/analyst/dashboard/{meta.case_id}")
        assert resp.status_code == 409

    def test_dashboard_success(self, client, case_store):
        """단위 메타 집계 — 파트가 여러 개인 단위도 한 번만 센다"""
        case_id = _ready(case_store, "대시보드 테스트")
        _write_units(
            case_id,
            [
                (
                    "em:1",
                    "email",
                    {
                        "participants": ["Alice", "Bob"],
                        "date": "2025-01-15T09:00:00+09:00",
                        "topics": ["감사"],
                    },
                ),
                (
                    "em:2",
                    "email",
                    {
                        "participants": ["Alice"],
                        "date": "2025-02-10T09:00:00+09:00",
                        "topics": ["계약"],
                    },
                ),
                (
                    "ch:3",
                    "chat",
                    {
                        "participants": ["Bob", "Charlie"],
                        "date": "2025-02-20T10:00:00+09:00",
                        "topics": ["회의"],
                    },
                ),
            ],
        )

        resp = client.get(f"/api/analyst/dashboard/{case_id}")
        assert resp.status_code == 200

        data = resp.json()
        assert data["case_id"] == case_id
        assert data["case_name"] == "대시보드 테스트"
        assert data["total_chunks"] == 3  # 단위 수 (파트 4개)
        assert data["source_type_counts"] == {"email": 2, "teams_chat": 1}
        assert len(data["timeline"]) == 2  # 2025-01, 2025-02
        assert data["timeline"][1]["teams_chat_count"] == 1
        assert len(data["top_topics"]) == 3

    def test_dashboard_empty_index(self, client, case_store):
        """빈 색인"""
        case_id = _ready(case_store, "빈 케이스")
        _write_units(case_id, [])
        resp = client.get(f"/api/analyst/dashboard/{case_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_chunks"] == 0
        assert data["source_type_counts"] == {}
        assert data["top_participants"] == []

    def test_dashboard_missing_index_needs_reindex(self, client, case_store):
        """READY인데 v4 색인이 없는 (v3) 케이스 → 409 재인덱싱"""
        case_id = _ready(case_store, "v3 케이스")
        resp = client.get(f"/api/analyst/dashboard/{case_id}")
        assert resp.status_code == 409
        assert "재인덱싱" in resp.json()["detail"]

    def test_dashboard_archived_case(self, client, case_store):
        """보관된 케이스도 조회 가능"""
        case_id = _ready(case_store, "보관 케이스")
        case_store.update_status(case_id, CaseStatus.ARCHIVED)
        _write_units(case_id, [])
        resp = client.get(f"/api/analyst/dashboard/{case_id}")
        assert resp.status_code == 200
