"""CaseStore 유닛 테스트 — 케이스 CRUD + 라이프사이클 (SQLite)"""

import tempfile
from pathlib import Path

import pytest

from src.cases.case_store import (
    CaseMetadata,
    CaseNotFoundError,
    CaseStatus,
    CaseStore,
    InvalidStatusTransitionError,
)


@pytest.fixture
def store():
    """임시 SQLite DB 기반 CaseStore"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db_url = f"sqlite:///{db_path}"

        from src.db.database import reset_globals
        reset_globals()

        yield CaseStore(db_url=db_url)

        reset_globals()


# === 생성 테스트 ===


class TestCreate:
    def test_create_basic(self, store: CaseStore):
        """기본 케이스 생성"""
        meta = store.create("감사 프로젝트 A")
        assert meta.name == "감사 프로젝트 A"
        assert meta.status == CaseStatus.CREATED
        assert len(meta.case_id) == 12
        assert meta.total_documents == 0
        assert meta.total_chunks == 0

    def test_create_with_paths(self, store: CaseStore):
        """데이터 소스 경로 포함 생성"""
        meta = store.create(
            name="프로젝트 B",
            description="2024 감사",
            pst_paths=["/data/pst/a.pst", "/data/pst/b.pst"],
            doc_paths=["/data/docs/reports"],
        )
        assert meta.description == "2024 감사"
        assert len(meta.pst_paths) == 2
        assert "/data/docs/reports" in meta.doc_paths

    def test_create_persists_to_db(self, store: CaseStore):
        """생성된 케이스가 DB에 저장되는지 확인"""
        meta = store.create("저장 테스트")
        loaded = store.get(meta.case_id)
        assert loaded.case_id == meta.case_id

    def test_create_multiple_unique_ids(self, store: CaseStore):
        """여러 케이스 생성 시 ID가 고유한지 확인"""
        ids = {store.create(f"케이스 {i}").case_id for i in range(5)}
        assert len(ids) == 5


# === 조회 테스트 ===


class TestGet:
    def test_get_existing(self, store: CaseStore):
        """생성된 케이스 조회"""
        created = store.create("조회 테스트")
        loaded = store.get(created.case_id)
        assert loaded.case_id == created.case_id
        assert loaded.name == "조회 테스트"
        assert loaded.status == CaseStatus.CREATED

    def test_get_not_found(self, store: CaseStore):
        """존재하지 않는 케이스 조회"""
        with pytest.raises(CaseNotFoundError):
            store.get("nonexistent_id")

    def test_list_empty(self, store: CaseStore):
        """빈 저장소 목록 조회"""
        assert store.list_all() == []

    def test_list_multiple(self, store: CaseStore):
        """여러 케이스 목록 조회 (생성일 내림차순)"""
        store.create("첫 번째")
        store.create("두 번째")
        store.create("세 번째")
        cases = store.list_all()
        assert len(cases) == 3
        # 최신 생성 순
        assert cases[0].name == "세 번째"

    def test_exists(self, store: CaseStore):
        """exists 메서드"""
        meta = store.create("존재 확인")
        assert store.exists(meta.case_id) is True
        assert store.exists("fake_id") is False


# === 상태 전이 테스트 ===


class TestStatusTransition:
    def test_created_to_indexing(self, store: CaseStore):
        """created → indexing"""
        meta = store.create("상태 테스트")
        updated = store.update_status(meta.case_id, CaseStatus.INDEXING)
        assert updated.status == CaseStatus.INDEXING

    def test_indexing_to_ready(self, store: CaseStore):
        """indexing → ready (indexed_at 타임스탬프 기록)"""
        meta = store.create("인덱싱 완료")
        store.update_status(meta.case_id, CaseStatus.INDEXING)
        updated = store.update_status(meta.case_id, CaseStatus.READY)
        assert updated.status == CaseStatus.READY
        assert updated.indexed_at is not None

    def test_ready_to_archived(self, store: CaseStore):
        """ready → archived"""
        meta = store.create("보관 테스트")
        store.update_status(meta.case_id, CaseStatus.INDEXING)
        store.update_status(meta.case_id, CaseStatus.READY)
        updated = store.update_status(meta.case_id, CaseStatus.ARCHIVED)
        assert updated.status == CaseStatus.ARCHIVED

    def test_ready_to_indexing_reindex(self, store: CaseStore):
        """ready → indexing (재인덱싱 / 추가 자료 유입)"""
        meta = store.create("재인덱싱")
        store.update_status(meta.case_id, CaseStatus.INDEXING)
        store.update_status(meta.case_id, CaseStatus.READY)
        updated = store.update_status(meta.case_id, CaseStatus.INDEXING)
        assert updated.status == CaseStatus.INDEXING

    def test_indexing_to_error(self, store: CaseStore):
        """indexing → error"""
        meta = store.create("에러 테스트")
        store.update_status(meta.case_id, CaseStatus.INDEXING)
        updated = store.update_status(meta.case_id, CaseStatus.ERROR)
        assert updated.status == CaseStatus.ERROR

    def test_error_to_indexing_retry(self, store: CaseStore):
        """error → indexing (재시도)"""
        meta = store.create("재시도")
        store.update_status(meta.case_id, CaseStatus.INDEXING)
        store.update_status(meta.case_id, CaseStatus.ERROR)
        updated = store.update_status(meta.case_id, CaseStatus.INDEXING)
        assert updated.status == CaseStatus.INDEXING

    def test_invalid_created_to_ready(self, store: CaseStore):
        """created → ready 불가 (인덱싱 없이 바로 ready 불가)"""
        meta = store.create("잘못된 전이")
        with pytest.raises(InvalidStatusTransitionError):
            store.update_status(meta.case_id, CaseStatus.READY)

    def test_invalid_archived_to_any(self, store: CaseStore):
        """archived → * 불가 (보관 상태에서 전이 불가)"""
        meta = store.create("보관 후 변경 불가")
        store.update_status(meta.case_id, CaseStatus.ARCHIVED)
        with pytest.raises(InvalidStatusTransitionError):
            store.update_status(meta.case_id, CaseStatus.CREATED)

    def test_not_found_status_update(self, store: CaseStore):
        """존재하지 않는 케이스 상태 변경"""
        with pytest.raises(CaseNotFoundError):
            store.update_status("fake_id", CaseStatus.INDEXING)


# === 데이터 소스 업데이트 테스트 ===


class TestUpdateDataSources:
    def test_add_pst_paths(self, store: CaseStore):
        """PST 경로 추가"""
        meta = store.create("소스 추가", pst_paths=["/a.pst"])
        updated = store.update_data_sources(meta.case_id, pst_paths=["/b.pst"])
        assert "/a.pst" in updated.pst_paths
        assert "/b.pst" in updated.pst_paths

    def test_add_doc_paths(self, store: CaseStore):
        """문서 경로 추가"""
        meta = store.create("문서 추가")
        updated = store.update_data_sources(meta.case_id, doc_paths=["/docs/reports"])
        assert "/docs/reports" in updated.doc_paths

    def test_no_duplicate_paths(self, store: CaseStore):
        """중복 경로는 추가되지 않음"""
        meta = store.create("중복 방지", pst_paths=["/a.pst"])
        updated = store.update_data_sources(meta.case_id, pst_paths=["/a.pst", "/b.pst"])
        assert updated.pst_paths.count("/a.pst") == 1
        assert "/b.pst" in updated.pst_paths

    def test_archived_case_cannot_update(self, store: CaseStore):
        """보관된 케이스는 데이터 소스 수정 불가"""
        meta = store.create("보관 케이스")
        store.update_status(meta.case_id, CaseStatus.ARCHIVED)
        with pytest.raises(InvalidStatusTransitionError):
            store.update_data_sources(meta.case_id, pst_paths=["/new.pst"])


# === 통계 업데이트 테스트 ===


class TestUpdateStats:
    def test_update_document_count(self, store: CaseStore):
        """문서 수 업데이트"""
        meta = store.create("통계 테스트")
        updated = store.update_stats(meta.case_id, total_documents=150)
        assert updated.total_documents == 150
        assert updated.total_chunks == 0  # 변경 안 됨

    def test_update_chunk_count(self, store: CaseStore):
        """청크 수 업데이트"""
        meta = store.create("청크 통계")
        updated = store.update_stats(meta.case_id, total_chunks=3000)
        assert updated.total_chunks == 3000

    def test_update_both(self, store: CaseStore):
        """문서 + 청크 동시 업데이트"""
        meta = store.create("동시 업데이트")
        updated = store.update_stats(meta.case_id, total_documents=50, total_chunks=1500)
        assert updated.total_documents == 50
        assert updated.total_chunks == 1500


# === 에러 상태 테스트 ===


class TestSetError:
    def test_set_error(self, store: CaseStore):
        """에러 메시지 기록"""
        meta = store.create("에러 기록")
        updated = store.set_error(meta.case_id, "파싱 실패: corrupted PST")
        assert updated.status == CaseStatus.ERROR
        assert "corrupted PST" in updated.error_message


# === 삭제 테스트 ===


class TestDelete:
    def test_delete_existing(self, store: CaseStore):
        """케이스 삭제"""
        meta = store.create("삭제 대상")
        store.delete(meta.case_id)
        assert not store.exists(meta.case_id)

    def test_delete_not_found(self, store: CaseStore):
        """존재하지 않는 케이스 삭제"""
        with pytest.raises(CaseNotFoundError):
            store.delete("nonexistent_id")

    def test_delete_removes_from_db(self, store: CaseStore):
        """삭제 시 DB에서 제거 확인"""
        meta = store.create("DB 삭제")
        store.delete(meta.case_id)
        with pytest.raises(CaseNotFoundError):
            store.get(meta.case_id)


# === 직렬화 라운드트립 ===


class TestSerialization:
    def test_roundtrip(self, store: CaseStore):
        """저장 후 로드 시 데이터 보존"""
        meta = store.create(
            name="라운드트립",
            description="직렬화 검증",
            pst_paths=["/a.pst"],
            doc_paths=["/docs"],
        )
        store.update_status(meta.case_id, CaseStatus.INDEXING)
        store.update_stats(meta.case_id, total_documents=10, total_chunks=500)
        store.update_status(meta.case_id, CaseStatus.READY)

        loaded = store.get(meta.case_id)
        assert loaded.name == "라운드트립"
        assert loaded.description == "직렬화 검증"
        assert loaded.status == CaseStatus.READY
        assert loaded.pst_paths == ["/a.pst"]
        assert loaded.doc_paths == ["/docs"]
        assert loaded.total_documents == 10
        assert loaded.total_chunks == 500
        assert loaded.indexed_at is not None


class TestRecoverIfStuck:
    """stuck indexing 자동 복구 — INDEXING 상태에서 N분 멈춰있으면 ERROR로 강제 전이"""

    def test_not_stuck_returns_none(self, store):
        """방금 indexing 시작한 케이스는 복구 대상 아님"""
        meta = store.create(name="stuck-test")
        store.update_status(meta.case_id, CaseStatus.INDEXING)
        # 방금 갱신됨 → timeout 미경과
        result = store.recover_if_stuck(meta.case_id, timeout_min=30)
        assert result is None
        # 상태는 그대로 INDEXING
        assert store.get(meta.case_id).status == CaseStatus.INDEXING

    def test_stuck_recovers_to_error(self, store):
        """timeout_min 경과 후 INDEXING이면 ERROR로 전이"""
        from datetime import datetime, timedelta

        from src.db.models import CaseModel

        meta = store.create(name="stuck-old")
        store.update_status(meta.case_id, CaseStatus.INDEXING)
        # updated_at을 강제로 1시간 전으로 변경
        with store._get_session() as session:
            row = session.get(CaseModel, meta.case_id)
            row.updated_at = datetime.now() - timedelta(hours=1)
            session.commit()

        result = store.recover_if_stuck(meta.case_id, timeout_min=30)
        assert result is not None
        assert result.status == CaseStatus.ERROR
        assert "비정상 종료" in result.error_message

        # DB에도 반영
        loaded = store.get(meta.case_id)
        assert loaded.status == CaseStatus.ERROR

    def test_non_indexing_state_returns_none(self, store):
        """READY 등 INDEXING이 아닌 상태는 복구 대상 아님"""
        from datetime import datetime, timedelta

        from src.db.models import CaseModel

        meta = store.create(name="ready-old")
        store.update_status(meta.case_id, CaseStatus.INDEXING)
        store.update_status(meta.case_id, CaseStatus.READY)
        # 오래된 케이스로 위장
        with store._get_session() as session:
            row = session.get(CaseModel, meta.case_id)
            row.updated_at = datetime.now() - timedelta(hours=2)
            session.commit()

        # READY 상태 → 복구 대상 아님
        result = store.recover_if_stuck(meta.case_id, timeout_min=30)
        assert result is None

    def test_nonexistent_case_returns_none(self, store):
        """존재하지 않는 케이스는 None 반환 (예외 X)"""
        result = store.recover_if_stuck("nonexistent_id", timeout_min=30)
        assert result is None
