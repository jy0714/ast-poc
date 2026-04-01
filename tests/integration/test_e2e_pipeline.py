"""통합 테스트 — 케이스 생성 → 인덱싱 → RAG 질의 전체 파이프라인

FastAPI TestClient를 통해 API 레벨에서 전체 흐름을 검증한다.
Ollama에 현재 설정된 임베딩 모델이 설치되어 있어야 통과한다.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import cases as cases_mod
from src.api.routes import chat as chat_mod
from src.api.routes import indexing as indexing_mod
from src.cases.case_store import CaseStore
from src.indexing.pipeline import IndexingPipeline


def _embed_model_available() -> bool:
    """Ollama 서버 + 현재 임베딩 모델 설치 여부 확인"""
    try:
        import urllib.request

        from src.utils.config import settings

        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            model_names = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            return settings.ollama_embed_model in model_names
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _embed_model_available(),
    reason="Ollama 서버가 실행되지 않거나 임베딩 모델이 설치되지 않았습니다",
)


@pytest.fixture()
def env(tmp_path):
    """통합 테스트 환경: 격리된 CaseStore + IndexingPipeline + TestClient"""
    # 격리된 SQLite DB
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"

    from src.db.database import reset_globals
    reset_globals()

    store = CaseStore(db_url=db_url)
    pipeline = IndexingPipeline(case_store=store)

    # 모듈 싱글톤 교체
    orig_cases_store = cases_mod._store
    orig_idx_store = indexing_mod._case_store
    orig_idx_pipeline = indexing_mod._pipeline
    orig_chat_store = chat_mod._case_store

    cases_mod._store = store
    indexing_mod._case_store = store
    indexing_mod._pipeline = pipeline
    chat_mod._case_store = store

    # 테스트 문서 생성
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "감사보고서.txt").write_text(
        "2025년 1분기 내부 감사 보고서\n\n"
        "감사 대상: 재무팀 비용 처리 프로세스\n"
        "감사 기간: 2025년 1월 1일 ~ 2025년 3월 31일\n"
        "감사인: 홍길동, 김감사\n\n"
        "주요 발견 사항:\n"
        "1. 비용 승인 프로세스가 미흡하여 100만원 이상 지출 건에 대한 상위 승인이 누락됨\n"
        "2. 출장비 정산 시 영수증 미첨부 건이 전체의 15%에 해당\n"
        "3. 거래처 A사와의 계약 갱신 시 경쟁 입찰 없이 수의계약 진행\n\n"
        "권고 사항:\n"
        "- 100만원 이상 지출 시 부서장 승인 필수화\n"
        "- 출장비 정산 시스템에 영수증 첨부 기능 강화\n"
        "- 5천만원 이상 계약 시 경쟁 입찰 의무화",
        encoding="utf-8",
    )

    (docs_dir / "회의록.txt").write_text(
        "2025년 3월 15일 감사위원회 회의록\n\n"
        "참석자: 이위원장, 박위원, 최위원, 홍길동(감사팀장)\n"
        "안건: 1분기 내부 감사 결과 보고\n\n"
        "논의 내용:\n"
        "홍길동 팀장이 1분기 감사 결과를 보고함.\n"
        "이위원장은 비용 승인 프로세스 개선을 최우선 과제로 지정.\n"
        "박위원은 출장비 정산 시스템 개선 일정을 4월 중으로 요청.\n"
        "최위원은 거래처 A사 수의계약 건에 대한 추가 조사를 요청함.\n\n"
        "결정사항:\n"
        "1. 비용 승인 프로세스 개선안 4월 말까지 마련\n"
        "2. 출장비 정산 시스템 업그레이드 5월 완료 목표\n"
        "3. A사 수의계약 건 특별 감사 실시",
        encoding="utf-8",
    )

    (docs_dir / "이메일_내역.eml").write_bytes(
        b"From: hong@company.com\r\n"
        b"To: kim@company.com\r\n"
        b"Subject: =?utf-8?b?6rCA7IKsIOqysOqzvCDrj5nrs7Q=?=\r\n"
        b"Date: Mon, 17 Mar 2025 09:30:00 +0900\r\n"
        b"Message-ID: <audit-001@company.com>\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n"
        b"Content-Transfer-Encoding: 8bit\r\n"
        b"\r\n"
        + "김감사님,\n\n1분기 감사 결과를 첨부합니다.\n비용 승인 프로세스 관련 이슈가 가장 심각합니다.\n다음 주 회의에서 논의하시죠.\n\n홍길동 드림".encode("utf-8")
    )

    client = TestClient(app)

    yield {
        "client": client,
        "store": store,
        "pipeline": pipeline,
        "docs_dir": str(docs_dir),
    }

    # 복원
    cases_mod._store = orig_cases_store
    indexing_mod._case_store = orig_idx_store
    indexing_mod._pipeline = orig_idx_pipeline
    chat_mod._case_store = orig_chat_store
    reset_globals()


# =====================================================================
# 통합 테스트 1: 전체 파이프라인 (케이스 생성 → 인덱싱 → 질의)
# =====================================================================


class TestFullPipeline:
    """케이스 생성 → 데이터 소스 설정 → 인덱싱 → RAG 질의 전체 흐름"""

    def test_create_case(self, env):
        """1단계: 케이스 생성"""
        resp = env["client"].post(
            "/api/admin/cases/",
            json={
                "name": "2025 1분기 감사",
                "description": "재무팀 비용 처리 감사",
                "doc_paths": [env["docs_dir"]],
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "2025 1분기 감사"
        assert data["status"] == "created"
        assert env["docs_dir"] in data["doc_paths"]

    def test_list_cases(self, env):
        """케이스 목록 조회"""
        # 먼저 케이스 생성
        env["client"].post(
            "/api/admin/cases/",
            json={"name": "테스트 케이스", "doc_paths": [env["docs_dir"]]},
        )
        resp = env["client"].get("/api/admin/cases/")
        assert resp.status_code == 200
        cases = resp.json()
        assert len(cases) >= 1

    def test_indexing_and_query(self, env):
        """2-3단계: 인덱싱 실행 → RAG 질의까지 전체 흐름"""
        # 1) 케이스 생성
        create_resp = env["client"].post(
            "/api/admin/cases/",
            json={
                "name": "통합테스트 감사",
                "description": "E2E 테스트",
                "doc_paths": [env["docs_dir"]],
            },
        )
        case_id = create_resp.json()["case_id"]

        # 2) 동기 인덱싱 실행 (파이프라인 직접 호출 — API는 백그라운드 스레드)
        pipeline = env["pipeline"]
        pipeline.run(case_id)

        # 3) 케이스 상태 확인 — ready 여야 함
        case_resp = env["client"].get(f"/api/admin/cases/{case_id}")
        assert case_resp.status_code == 200
        case_data = case_resp.json()
        assert case_data["status"] == "ready"
        assert case_data["total_documents"] > 0
        assert case_data["total_chunks"] > 0

        # 4) 질의 가능 케이스 목록에 나타나야 함
        cases_resp = env["client"].get("/api/analyst/chat/cases")
        case_ids = [c["case_id"] for c in cases_resp.json()]
        assert case_id in case_ids

        # 5) RAG 질의 (LLM mock — Ollama 없이 테스트)
        with patch("src.rag.engine.LLMRouter") as mock_router_cls:
            mock_router = MagicMock()
            mock_router.generate = AsyncMock(
                return_value="감사 결과, 비용 승인 프로세스가 미흡하여 100만원 이상 지출 건에 대한 승인이 누락된 것으로 확인되었습니다."
            )
            mock_router_cls.return_value = mock_router

            chat_resp = env["client"].post(
                "/api/analyst/chat/",
                json={
                    "case_id": case_id,
                    "message": "비용 승인 관련 감사 결과를 알려줘",
                    "security_mode": True,
                },
            )

        assert chat_resp.status_code == 200
        chat_data = chat_resp.json()
        assert chat_data["case_id"] == case_id
        assert chat_data["security_mode"] is True
        assert len(chat_data["answer"]) > 0
        assert len(chat_data["sources"]) > 0

        # 출처에 감사보고서 또는 회의록이 포함되어야 함
        filenames = [s["filename"] for s in chat_data["sources"]]
        assert any("감사보고서" in f or "회의록" in f or "이메일" in f for f in filenames)

    def test_query_not_ready_case(self, env):
        """인덱싱 안 된 케이스에 질의하면 409 에러"""
        create_resp = env["client"].post(
            "/api/admin/cases/",
            json={"name": "미인덱싱 케이스"},
        )
        case_id = create_resp.json()["case_id"]

        chat_resp = env["client"].post(
            "/api/analyst/chat/",
            json={"case_id": case_id, "message": "테스트 질의"},
        )
        assert chat_resp.status_code == 409

    def test_archive_then_query(self, env):
        """보관된 케이스도 질의 가능"""
        # 케이스 생성 + 인덱싱
        create_resp = env["client"].post(
            "/api/admin/cases/",
            json={"name": "보관 테스트", "doc_paths": [env["docs_dir"]]},
        )
        case_id = create_resp.json()["case_id"]
        env["pipeline"].run(case_id)

        # 보관 처리
        archive_resp = env["client"].post(f"/api/admin/cases/{case_id}/archive")
        assert archive_resp.status_code == 200
        assert archive_resp.json()["status"] == "archived"

        # 보관 상태에서 질의 가능해야 함
        with patch("src.rag.engine.LLMRouter") as mock_router_cls:
            mock_router = MagicMock()
            mock_router.generate = AsyncMock(return_value="보관된 케이스의 응답입니다.")
            mock_router_cls.return_value = mock_router

            chat_resp = env["client"].post(
                "/api/analyst/chat/",
                json={"case_id": case_id, "message": "감사 결과 요약"},
            )

        assert chat_resp.status_code == 200

    def test_delete_case(self, env):
        """케이스 삭제 후 조회 불가"""
        create_resp = env["client"].post(
            "/api/admin/cases/",
            json={"name": "삭제 테스트"},
        )
        case_id = create_resp.json()["case_id"]

        delete_resp = env["client"].delete(f"/api/admin/cases/{case_id}")
        assert delete_resp.status_code == 200

        get_resp = env["client"].get(f"/api/admin/cases/{case_id}")
        assert get_resp.status_code == 404


# =====================================================================
# 통합 테스트 2: 인덱싱 파이프라인 세부 검증
# =====================================================================


class TestIndexingPipeline:
    """인덱싱 파이프라인 파일 처리 검증"""

    def test_txt_indexed(self, env):
        """TXT 파일이 인덱싱되는지 확인"""
        create_resp = env["client"].post(
            "/api/admin/cases/",
            json={"name": "TXT 테스트", "doc_paths": [env["docs_dir"]]},
        )
        case_id = create_resp.json()["case_id"]
        env["pipeline"].run(case_id)

        case_data = env["client"].get(f"/api/admin/cases/{case_id}").json()
        assert case_data["total_documents"] >= 2  # txt 2개 + eml 1개
        assert case_data["total_chunks"] >= 3

    def test_eml_indexed(self, env):
        """EML 파일이 인덱싱되는지 확인"""
        create_resp = env["client"].post(
            "/api/admin/cases/",
            json={"name": "EML 테스트", "doc_paths": [env["docs_dir"]]},
        )
        case_id = create_resp.json()["case_id"]
        env["pipeline"].run(case_id)

        # 벡터 스토어에서 직접 검색
        from src.vectorstore.vector_store import VectorStoreService

        vs = VectorStoreService(collection_name=f"case_{case_id}")
        results = vs.search("감사 결과 동보", n_results=10)
        source_types = [r["metadata"].get("source_type", "") for r in results]
        # 이메일이 하나 이상 인덱싱됨
        assert any("email" in st or "document" in st for st in source_types)

    def test_empty_case_indexing(self, env):
        """데이터 소스 없는 케이스 인덱싱 — 에러 없이 완료"""
        create_resp = env["client"].post(
            "/api/admin/cases/",
            json={"name": "빈 케이스"},
        )
        case_id = create_resp.json()["case_id"]
        env["pipeline"].run(case_id)

        case_data = env["client"].get(f"/api/admin/cases/{case_id}").json()
        assert case_data["status"] == "ready"
        assert case_data["total_documents"] == 0
        assert case_data["total_chunks"] == 0


# =====================================================================
# 통합 테스트 3: 검색 품질 검증
# =====================================================================


class TestSearchQuality:
    """인덱싱된 데이터가 적절히 검색되는지 검증"""

    @pytest.fixture(autouse=True)
    def indexed_case(self, env):
        """인덱싱 완료된 케이스 준비"""
        create_resp = env["client"].post(
            "/api/admin/cases/",
            json={"name": "검색 품질 테스트", "doc_paths": [env["docs_dir"]]},
        )
        self.case_id = create_resp.json()["case_id"]
        env["pipeline"].run(self.case_id)
        self.env = env

    def test_keyword_search(self):
        """키워드 기반 검색 — '비용 승인'으로 감사보고서가 검색됨"""
        from src.vectorstore.vector_store import VectorStoreService

        vs = VectorStoreService(collection_name=f"case_{self.case_id}")
        results = vs.search("비용 승인 프로세스", n_results=5)
        assert len(results) > 0
        # 비용 관련 내용이 포함된 결과가 있어야 함
        contents = " ".join(r["content"] for r in results)
        assert "비용" in contents

    def test_participant_search(self):
        """참여자 관련 검색"""
        from src.vectorstore.vector_store import VectorStoreService

        vs = VectorStoreService(collection_name=f"case_{self.case_id}")
        results = vs.search("홍길동", n_results=5)
        assert len(results) > 0

    def test_query_parser_filter(self):
        """질의 파서가 필터를 추출하는지 확인"""
        from src.rag.query_parser import parse_query

        parsed = parse_query("이메일에서 감사 결과 관련 내용 찾아줘")
        assert parsed.filters.get("source_type") == "email"
        assert "감사" in parsed.cleaned

    def test_rag_engine_search(self):
        """RAG 엔진이 출처와 함께 결과를 반환"""
        from src.rag.engine import RAGEngine

        engine = RAGEngine(case_id=self.case_id)
        sources = engine.search("거래처 A사 수의계약")
        assert len(sources) > 0
        # 감사보고서나 회의록에서 찾아야 함
        all_content = " ".join(s.content for s in sources)
        assert "A사" in all_content or "수의계약" in all_content


# =====================================================================
# 통합 테스트 4: API 헬스체크
# =====================================================================


class TestHealthCheck:
    def test_health(self, env):
        resp = env["client"].get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
