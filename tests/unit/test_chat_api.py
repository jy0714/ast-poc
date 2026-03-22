"""Chat API 엔드포인트 테스트"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import chat as chat_module
from src.cases.case_store import CaseStatus, CaseStore


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def store(tmpdir):
    """테스트용 CaseStore 교체"""
    temp_store = CaseStore(base_dir=tmpdir / "cases")
    original = chat_module._case_store
    chat_module._case_store = temp_store
    yield temp_store
    chat_module._case_store = original


@pytest.fixture
def client(store):
    return TestClient(app)


def _create_ready_case(store: CaseStore, name: str = "테스트 케이스") -> str:
    """ready 상태 케이스 생성"""
    meta = store.create(name)
    store.update_status(meta.case_id, CaseStatus.INDEXING)
    store.update_stats(meta.case_id, total_documents=10, total_chunks=100)
    store.update_status(meta.case_id, CaseStatus.READY)
    return meta.case_id


# === 케이스 목록 API ===


class TestGetCases:
    def test_empty_cases(self, client):
        """GET /cases — 빈 목록"""
        resp = client.get("/api/analyst/chat/cases")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_only_ready_cases(self, client, store):
        """GET /cases — ready/archived만 반환"""
        # ready 케이스
        ready_id = _create_ready_case(store, "준비됨")
        # created 케이스 (미포함)
        store.create("생성만")

        resp = client.get("/api/analyst/chat/cases")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["case_id"] == ready_id
        assert data[0]["name"] == "준비됨"
        assert data[0]["total_documents"] == 10

    def test_archived_case_included(self, client, store):
        """GET /cases — archived 케이스도 포함"""
        case_id = _create_ready_case(store, "보관 케이스")
        store.update_status(case_id, CaseStatus.ARCHIVED)

        resp = client.get("/api/analyst/chat/cases")
        assert len(resp.json()) == 1


# === 채팅 API ===


class TestChatAPI:
    def test_chat_case_not_found(self, client):
        """POST / — 404"""
        resp = client.post("/api/analyst/chat/", json={
            "case_id": "nonexistent",
            "message": "질문",
        })
        assert resp.status_code == 404

    def test_chat_case_not_ready(self, client, store):
        """POST / — 인덱싱 안 된 케이스 409"""
        meta = store.create("미완료")
        resp = client.post("/api/analyst/chat/", json={
            "case_id": meta.case_id,
            "message": "질문",
        })
        assert resp.status_code == 409

    def test_chat_empty_message(self, client, store):
        """POST / — 빈 메시지 400"""
        case_id = _create_ready_case(store)
        resp = client.post("/api/analyst/chat/", json={
            "case_id": case_id,
            "message": "   ",
        })
        assert resp.status_code == 400

    @patch("src.api.routes.chat.RAGEngine")
    def test_chat_success(self, mock_engine_cls, client, store):
        """POST / — 정상 질의"""
        case_id = _create_ready_case(store)

        # RAGEngine 모킹
        mock_engine = AsyncMock()
        mock_engine.query.return_value = MagicMock(
            answer="테스트 답변입니다.",
            sources=[
                MagicMock(
                    content="출처 내용",
                    source_type="email",
                    filename="test.eml",
                    date="2025-03-15",
                    participants=["홍길동"],
                    subject="회의록",
                    score=0.85,
                    search_method="hybrid",
                ),
            ],
            secure_mode=True,
            case_id=case_id,
        )
        mock_engine_cls.return_value = mock_engine

        resp = client.post("/api/analyst/chat/", json={
            "case_id": case_id,
            "message": "비용 관련 내용 알려줘",
            "security_mode": True,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "테스트 답변입니다."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source_type"] == "email"
        assert data["case_id"] == case_id
        assert data["security_mode"] is True

    @patch("src.api.routes.chat.RAGEngine")
    def test_chat_with_filters(self, mock_engine_cls, client, store):
        """POST / — 필터 포함 질의"""
        case_id = _create_ready_case(store)

        mock_engine = AsyncMock()
        mock_engine.query.return_value = MagicMock(
            answer="답변",
            sources=[],
            secure_mode=True,
            case_id=case_id,
        )
        mock_engine_cls.return_value = mock_engine

        resp = client.post("/api/analyst/chat/", json={
            "case_id": case_id,
            "message": "이메일 내용 찾아줘",
            "filters": {"source_type": "email"},
        })

        assert resp.status_code == 200
        # 필터가 전달되었는지 확인
        mock_engine.query.assert_called_once()
        _, kwargs = mock_engine.query.call_args
        assert kwargs["filters"] == {"source_type": "email"}


# === 스트리밍 API ===


class TestStreamAPI:
    def test_stream_case_not_found(self, client):
        """POST /stream — 404"""
        resp = client.post("/api/analyst/chat/stream", json={
            "case_id": "fake",
            "message": "질문",
        })
        assert resp.status_code == 404

    def test_stream_case_not_ready(self, client, store):
        """POST /stream — 미준비 케이스 409"""
        meta = store.create("미완료")
        resp = client.post("/api/analyst/chat/stream", json={
            "case_id": meta.case_id,
            "message": "질문",
        })
        assert resp.status_code == 409
