"""케이스 저장소 — SQLite 기반 케이스 CRUD + 라이프사이클 관리

케이스 라이프사이클:
    created → indexing → ready → archived
                ↑          |
                └── (추가 자료 유입)

저장소:
    data/ast.db (cases 테이블) — SQLAlchemy ORM
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CaseStatus(str, Enum):
    """케이스 상태"""

    CREATED = "created"
    INDEXING = "indexing"
    READY = "ready"
    ARCHIVED = "archived"
    ERROR = "error"


# 상태 전이 규칙: 현재 상태 → 허용되는 다음 상태
_VALID_TRANSITIONS: dict[CaseStatus, set[CaseStatus]] = {
    CaseStatus.CREATED: {CaseStatus.INDEXING, CaseStatus.ARCHIVED},
    CaseStatus.INDEXING: {CaseStatus.READY, CaseStatus.ERROR, CaseStatus.CREATED},
    CaseStatus.READY: {CaseStatus.INDEXING, CaseStatus.ARCHIVED},
    CaseStatus.ERROR: {CaseStatus.INDEXING, CaseStatus.CREATED, CaseStatus.ARCHIVED},
    CaseStatus.ARCHIVED: set(),  # 보관 상태에서는 전이 불가
}


@dataclass
class CaseMetadata:
    """케이스 메타데이터"""

    case_id: str
    name: str
    description: str
    status: CaseStatus
    created_at: datetime
    updated_at: datetime
    pst_paths: list[str] = field(default_factory=list)
    doc_paths: list[str] = field(default_factory=list)
    total_documents: int = 0
    total_chunks: int = 0
    indexed_at: datetime | None = None
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화용 딕셔너리 변환"""
        data = {
            "case_id": self.case_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "pst_paths": self.pst_paths,
            "doc_paths": self.doc_paths,
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "error_message": self.error_message,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseMetadata:
        """딕셔너리에서 CaseMetadata 생성"""
        return cls(
            case_id=data["case_id"],
            name=data["name"],
            description=data.get("description", ""),
            status=CaseStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            pst_paths=data.get("pst_paths", []),
            doc_paths=data.get("doc_paths", []),
            total_documents=data.get("total_documents", 0),
            total_chunks=data.get("total_chunks", 0),
            indexed_at=datetime.fromisoformat(data["indexed_at"]) if data.get("indexed_at") else None,
            error_message=data.get("error_message", ""),
        )


class CaseNotFoundError(Exception):
    """케이스를 찾을 수 없을 때 발생"""


class InvalidStatusTransitionError(Exception):
    """허용되지 않는 상태 전이 시 발생"""


def _model_to_metadata(row: Any) -> CaseMetadata:
    """CaseModel → CaseMetadata 변환"""
    return CaseMetadata(
        case_id=row.case_id,
        name=row.name,
        description=row.description,
        status=CaseStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
        pst_paths=json.loads(row.pst_paths) if row.pst_paths else [],
        doc_paths=json.loads(row.doc_paths) if row.doc_paths else [],
        total_documents=row.total_documents,
        total_chunks=row.total_chunks,
        indexed_at=row.indexed_at,
        error_message=row.error_message,
    )


class CaseStore:
    """SQLite 기반 케이스 저장소

    사용법:
        store = CaseStore()
        case = store.create("프로젝트 감사", pst_paths=["/data/pst/proj.pst"])
        cases = store.list_all()
        store.update_status(case.case_id, CaseStatus.INDEXING)
        store.delete(case.case_id)
    """

    def __init__(self, db_url: str | None = None) -> None:
        """CaseStore 초기화

        Args:
            db_url: 커스텀 DB URL (테스트용). None이면 기본 data/ast.db 사용.
        """
        from src.db.database import init_db, get_session_factory

        self._engine = init_db(db_url)
        self._session_factory = get_session_factory(self._engine)

    def _get_session(self):  # noqa: ANN202
        """새 DB 세션 반환"""
        return self._session_factory()

    def create(
        self,
        name: str,
        description: str = "",
        pst_paths: list[str] | None = None,
        doc_paths: list[str] | None = None,
    ) -> CaseMetadata:
        """새 케이스 생성

        Args:
            name: 케이스 이름
            description: 설명
            pst_paths: PST 파일 경로 리스트
            doc_paths: 문서 폴더 경로 리스트

        Returns:
            생성된 CaseMetadata
        """
        from src.db.models import CaseModel

        case_id = uuid.uuid4().hex[:12]
        now = datetime.now()

        row = CaseModel(
            case_id=case_id,
            name=name,
            description=description,
            status=CaseStatus.CREATED.value,
            created_at=now,
            updated_at=now,
            pst_paths=json.dumps(pst_paths or [], ensure_ascii=False),
            doc_paths=json.dumps(doc_paths or [], ensure_ascii=False),
        )

        with self._get_session() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            meta = _model_to_metadata(row)

        logger.info(f"케이스 생성: {case_id} ({name})")
        return meta

    def get(self, case_id: str) -> CaseMetadata:
        """케이스 조회

        Args:
            case_id: 케이스 ID

        Returns:
            CaseMetadata

        Raises:
            CaseNotFoundError: 케이스가 존재하지 않음
        """
        from src.db.models import CaseModel

        with self._get_session() as session:
            row = session.get(CaseModel, case_id)
            if row is None:
                raise CaseNotFoundError(f"케이스를 찾을 수 없습니다: {case_id}")
            return _model_to_metadata(row)

    def list_all(self) -> list[CaseMetadata]:
        """전체 케이스 목록 조회 (생성일 내림차순)"""
        from src.db.models import CaseModel

        with self._get_session() as session:
            rows = session.query(CaseModel).order_by(CaseModel.created_at.desc()).all()
            return [_model_to_metadata(r) for r in rows]

    def update_status(self, case_id: str, new_status: CaseStatus) -> CaseMetadata:
        """케이스 상태 변경 (라이프사이클 전이 규칙 검증)

        Args:
            case_id: 케이스 ID
            new_status: 변경할 상태

        Returns:
            업데이트된 CaseMetadata

        Raises:
            CaseNotFoundError: 케이스가 존재하지 않음
            InvalidStatusTransitionError: 허용되지 않는 상태 전이
        """
        from src.db.models import CaseModel

        with self._get_session() as session:
            row = session.get(CaseModel, case_id)
            if row is None:
                raise CaseNotFoundError(f"케이스를 찾을 수 없습니다: {case_id}")

            current = CaseStatus(row.status)
            allowed = _VALID_TRANSITIONS.get(current, set())
            if new_status not in allowed:
                raise InvalidStatusTransitionError(
                    f"상태 전이 불가: {current.value} → {new_status.value} "
                    f"(허용: {', '.join(s.value for s in allowed) or '없음'})"
                )

            row.status = new_status.value
            row.updated_at = datetime.now()

            if new_status == CaseStatus.READY:
                row.indexed_at = datetime.now()

            session.commit()
            session.refresh(row)
            meta = _model_to_metadata(row)

        logger.info(f"케이스 상태 변경: {case_id} → {new_status.value}")
        return meta

    def recover_if_stuck(
        self, case_id: str, timeout_min: int = 30
    ) -> CaseMetadata | None:
        """INDEXING 상태에서 멈춰있는 케이스를 ERROR로 강제 복구

        프로세스 kill / OOM / 정전 등으로 인덱싱이 비정상 종료된 후 케이스 상태가
        INDEXING으로 영구 고착되는 것을 방지. updated_at이 timeout_min 이상 갱신
        없으면 비정상 종료로 판정하고 ERROR로 전이 (라이프사이클 검증 우회).

        Args:
            case_id: 케이스 ID
            timeout_min: stuck 판정 임계값 (분)

        Returns:
            복구된 CaseMetadata. 복구 대상이 아니면 None.
        """
        from src.db.models import CaseModel

        with self._get_session() as session:
            row = session.get(CaseModel, case_id)
            if row is None:
                return None

            if CaseStatus(row.status) != CaseStatus.INDEXING:
                return None

            # updated_at 이후 경과 시간 체크
            elapsed_sec = (datetime.now() - row.updated_at).total_seconds()
            if elapsed_sec < timeout_min * 60:
                return None

            # 비정상 종료 판정 → 전이 검증 우회하고 ERROR로 강제
            row.status = CaseStatus.ERROR.value
            row.updated_at = datetime.now()
            row.error_message = (
                f"비정상 종료로 자동 복구됨 (INDEXING 상태에서 "
                f"{elapsed_sec / 60:.0f}분 멈춤)"
            )
            session.commit()
            session.refresh(row)
            meta = _model_to_metadata(row)

        logger.warning(
            f"stuck indexing 자동 복구: {case_id} "
            f"(updated_at 이후 {elapsed_sec / 60:.0f}분 경과 → ERROR)"
        )
        return meta

    def update_data_sources(
        self,
        case_id: str,
        pst_paths: list[str] | None = None,
        doc_paths: list[str] | None = None,
    ) -> CaseMetadata:
        """데이터 소스 경로 추가

        Args:
            case_id: 케이스 ID
            pst_paths: 추가할 PST 파일 경로
            doc_paths: 추가할 문서 폴더 경로

        Returns:
            업데이트된 CaseMetadata
        """
        from src.db.models import CaseModel

        with self._get_session() as session:
            row = session.get(CaseModel, case_id)
            if row is None:
                raise CaseNotFoundError(f"케이스를 찾을 수 없습니다: {case_id}")

            if CaseStatus(row.status) == CaseStatus.ARCHIVED:
                raise InvalidStatusTransitionError("보관된 케이스는 수정할 수 없습니다")

            current_pst = json.loads(row.pst_paths) if row.pst_paths else []
            current_doc = json.loads(row.doc_paths) if row.doc_paths else []

            if pst_paths:
                for p in pst_paths:
                    if p not in current_pst:
                        current_pst.append(p)

            if doc_paths:
                for p in doc_paths:
                    if p not in current_doc:
                        current_doc.append(p)

            row.pst_paths = json.dumps(current_pst, ensure_ascii=False)
            row.doc_paths = json.dumps(current_doc, ensure_ascii=False)
            row.updated_at = datetime.now()

            session.commit()
            session.refresh(row)
            meta = _model_to_metadata(row)

        logger.info(f"데이터 소스 업데이트: {case_id}")
        return meta

    def update_stats(
        self,
        case_id: str,
        total_documents: int | None = None,
        total_chunks: int | None = None,
    ) -> CaseMetadata:
        """인덱싱 통계 업데이트

        Args:
            case_id: 케이스 ID
            total_documents: 총 문서 수
            total_chunks: 총 청크 수
        """
        from src.db.models import CaseModel

        with self._get_session() as session:
            row = session.get(CaseModel, case_id)
            if row is None:
                raise CaseNotFoundError(f"케이스를 찾을 수 없습니다: {case_id}")

            if total_documents is not None:
                row.total_documents = total_documents
            if total_chunks is not None:
                row.total_chunks = total_chunks

            row.updated_at = datetime.now()
            session.commit()
            session.refresh(row)
            return _model_to_metadata(row)

    def set_error(self, case_id: str, error_message: str) -> CaseMetadata:
        """케이스를 에러 상태로 전환

        Args:
            case_id: 케이스 ID
            error_message: 에러 메시지
        """
        from src.db.models import CaseModel

        with self._get_session() as session:
            row = session.get(CaseModel, case_id)
            if row is None:
                raise CaseNotFoundError(f"케이스를 찾을 수 없습니다: {case_id}")

            row.status = CaseStatus.ERROR.value
            row.error_message = error_message
            row.updated_at = datetime.now()

            session.commit()
            session.refresh(row)
            meta = _model_to_metadata(row)

        logger.error(f"케이스 에러: {case_id} — {error_message}")
        return meta

    def delete(self, case_id: str) -> None:
        """케이스 삭제

        Args:
            case_id: 케이스 ID

        Raises:
            CaseNotFoundError: 케이스가 존재하지 않음
        """
        from src.db.models import CaseModel

        with self._get_session() as session:
            row = session.get(CaseModel, case_id)
            if row is None:
                raise CaseNotFoundError(f"케이스를 찾을 수 없습니다: {case_id}")

            session.delete(row)
            session.commit()

        logger.info(f"케이스 삭제 완료: {case_id}")

    def exists(self, case_id: str) -> bool:
        """케이스 존재 여부 확인"""
        from src.db.models import CaseModel

        with self._get_session() as session:
            return session.get(CaseModel, case_id) is not None
