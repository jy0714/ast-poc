"""데이터베이스 엔진/세션 관리

SQLite + SQLAlchemy 비동기 지원.
앱 시작 시 init_db()로 테이블 자동 생성.
향후 PostgreSQL 전환 시 DB_URL만 변경하면 됨.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# DB 경로
_DB_DIR = settings.project_root / "data"
_DB_PATH = _DB_DIR / "ast.db"
_DB_URL = f"sqlite:///{_DB_PATH}"

# 동기 엔진 (SQLite는 동기로도 충분, FastAPI에서 threadpool 사용)
_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection: object, connection_record: object) -> None:
    """SQLite WAL 모드 + FK 강제 활성화"""
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine(db_url: str | None = None) -> Engine:
    """SQLAlchemy 엔진 반환 (싱글톤)"""
    global _engine
    if _engine is None:
        url = db_url or _DB_URL
        # DB 디렉토리 생성
        if url.startswith("sqlite:///"):
            db_path = Path(url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(url, echo=False)
    return _engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """세션 팩토리 반환 (싱글톤)"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=engine or get_engine(), expire_on_commit=False)
    return _SessionLocal


def get_session() -> Session:
    """새 세션 생성"""
    factory = get_session_factory()
    return factory()


def init_db(db_url: str | None = None) -> Engine:
    """DB 초기화 — 테이블 자동 생성

    Args:
        db_url: 커스텀 DB URL (테스트용). None이면 기본 경로 사용.

    Returns:
        생성된 Engine
    """
    global _engine, _SessionLocal

    if db_url:
        # 커스텀 URL이면 싱글톤 리셋
        _engine = None
        _SessionLocal = None

    engine = get_engine(db_url)
    Base.metadata.create_all(bind=engine)
    _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    logger.info(f"DB 초기화 완료: {engine.url}")
    return engine


def reset_globals() -> None:
    """싱글톤 리셋 (테스트용)"""
    global _engine, _SessionLocal
    if _engine:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
