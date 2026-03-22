"""케이스 저장소 — JSON 파일 기반 케이스 CRUD + 라이프사이클 관리

케이스 라이프사이클:
    created → indexing → ready → archived
                ↑          |
                └── (추가 자료 유입)

저장 구조:
    data/cases/{case_id}/meta.json   — 케이스 메타데이터
    data/cases/{case_id}/            — 케이스별 디렉토리 (향후 인덱싱 산출물 저장)
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.utils.config import settings
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


class CaseStore:
    """JSON 파일 기반 케이스 저장소

    사용법:
        store = CaseStore()
        case = store.create("프로젝트 감사", pst_paths=["/data/pst/proj.pst"])
        cases = store.list_all()
        store.update_status(case.case_id, CaseStatus.INDEXING)
        store.delete(case.case_id)
    """

    META_FILENAME = "meta.json"

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """CaseStore 초기화

        Args:
            base_dir: 케이스 저장 루트 디렉토리 (기본: data/cases)
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = settings.project_root / "data" / "cases"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _case_dir(self, case_id: str) -> Path:
        """케이스 디렉토리 경로"""
        return self.base_dir / case_id

    def _meta_path(self, case_id: str) -> Path:
        """케이스 메타데이터 파일 경로"""
        return self._case_dir(case_id) / self.META_FILENAME

    def _save(self, meta: CaseMetadata) -> None:
        """메타데이터를 JSON 파일로 저장"""
        path = self._meta_path(meta.case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self, case_id: str) -> CaseMetadata:
        """JSON 파일에서 메타데이터 로드"""
        path = self._meta_path(case_id)
        if not path.exists():
            raise CaseNotFoundError(f"케이스를 찾을 수 없습니다: {case_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return CaseMetadata.from_dict(data)

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
        case_id = uuid.uuid4().hex[:12]
        now = datetime.now()

        meta = CaseMetadata(
            case_id=case_id,
            name=name,
            description=description,
            status=CaseStatus.CREATED,
            created_at=now,
            updated_at=now,
            pst_paths=pst_paths or [],
            doc_paths=doc_paths or [],
        )

        self._save(meta)
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
        return self._load(case_id)

    def list_all(self) -> list[CaseMetadata]:
        """전체 케이스 목록 조회 (생성일 내림차순)"""
        cases: list[CaseMetadata] = []

        if not self.base_dir.exists():
            return cases

        for case_dir in self.base_dir.iterdir():
            if not case_dir.is_dir():
                continue
            meta_path = case_dir / self.META_FILENAME
            if meta_path.exists():
                try:
                    cases.append(self._load(case_dir.name))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"케이스 메타데이터 로드 실패: {case_dir.name} ({e})")

        # 생성일 내림차순 정렬
        cases.sort(key=lambda c: c.created_at, reverse=True)
        return cases

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
        meta = self._load(case_id)

        allowed = _VALID_TRANSITIONS.get(meta.status, set())
        if new_status not in allowed:
            raise InvalidStatusTransitionError(
                f"상태 전이 불가: {meta.status.value} → {new_status.value} "
                f"(허용: {', '.join(s.value for s in allowed) or '없음'})"
            )

        meta.status = new_status
        meta.updated_at = datetime.now()

        if new_status == CaseStatus.READY:
            meta.indexed_at = datetime.now()

        self._save(meta)
        logger.info(f"케이스 상태 변경: {case_id} → {new_status.value}")
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
        meta = self._load(case_id)

        if meta.status == CaseStatus.ARCHIVED:
            raise InvalidStatusTransitionError("보관된 케이스는 수정할 수 없습니다")

        if pst_paths:
            for p in pst_paths:
                if p not in meta.pst_paths:
                    meta.pst_paths.append(p)

        if doc_paths:
            for p in doc_paths:
                if p not in meta.doc_paths:
                    meta.doc_paths.append(p)

        meta.updated_at = datetime.now()
        self._save(meta)
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
        meta = self._load(case_id)

        if total_documents is not None:
            meta.total_documents = total_documents
        if total_chunks is not None:
            meta.total_chunks = total_chunks

        meta.updated_at = datetime.now()
        self._save(meta)
        return meta

    def set_error(self, case_id: str, error_message: str) -> CaseMetadata:
        """케이스를 에러 상태로 전환

        Args:
            case_id: 케이스 ID
            error_message: 에러 메시지
        """
        meta = self._load(case_id)
        meta.status = CaseStatus.ERROR
        meta.error_message = error_message
        meta.updated_at = datetime.now()
        self._save(meta)
        logger.error(f"케이스 에러: {case_id} — {error_message}")
        return meta

    def delete(self, case_id: str) -> None:
        """케이스 삭제 (디렉토리 전체 삭제)

        Args:
            case_id: 케이스 ID

        Raises:
            CaseNotFoundError: 케이스가 존재하지 않음
        """
        case_dir = self._case_dir(case_id)
        if not case_dir.exists():
            raise CaseNotFoundError(f"케이스를 찾을 수 없습니다: {case_id}")

        shutil.rmtree(case_dir)
        logger.info(f"케이스 삭제 완료: {case_id}")

    def exists(self, case_id: str) -> bool:
        """케이스 존재 여부 확인"""
        return self._meta_path(case_id).exists()
