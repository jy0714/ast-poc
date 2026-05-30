"""데이터베이스 엔진/세션 관리

SQLite + SQLAlchemy 동기/비동기 양쪽 지원.

- 동기 (`get_session`): 기존 라우터/스크립트가 사용. 단순하고 안정적.
- 비동기 (`get_async_session`): FastAPI 핸들러에서 await 사용 시 권장.
  PoC 단계에서는 인프라만 깔고 점진적으로 라우터 전환.

향후 PostgreSQL 전환 시 DB_URL/AsyncDB_URL만 변경하면 됨.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# DB 경로
_DB_DIR = settings.project_root / "data"
_DB_PATH = _DB_DIR / "ast.db"
_DB_URL = f"sqlite:///{_DB_PATH}"
_ASYNC_DB_URL = f"sqlite+aiosqlite:///{_DB_PATH}"

# 동기 엔진 (SQLite는 동기로도 충분, FastAPI에서 threadpool 사용)
_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None

# 비동기 엔진 (FastAPI async 핸들러에서 사용)
_async_engine: AsyncEngine | None = None
_AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None


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
    _apply_lightweight_migrations(engine)
    _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    logger.info(f"DB 초기화 완료: {engine.url}")
    return engine


def _apply_lightweight_migrations(engine: Engine) -> None:
    """기존 SQLite DB에 누락된 컬럼을 ALTER TABLE로 추가

    Alembic 도입 전 PoC용. SQLAlchemy create_all은 새 테이블만 생성하고 기존
    테이블의 컬럼 추가는 처리하지 않으므로, 모델에 새 컬럼이 추가됐을 때
    기존 운영 DB가 깨지지 않도록 SQLite ADD COLUMN을 호출.
    """
    # (테이블, 컬럼명, SQL 정의) 튜플 — 모델 변경 시 이 리스트만 갱신
    expected_columns: list[tuple[str, str, str]] = [
        ("chat_sources", "sender", "VARCHAR(500) DEFAULT ''"),
        ("chat_sources", "recipients", "TEXT DEFAULT '[]'"),
        ("chat_sources", "cc", "TEXT DEFAULT '[]'"),
        ("chat_sources", "attachments", "TEXT DEFAULT '[]'"),
        ("chat_sources", "message_id", "VARCHAR(500) DEFAULT ''"),
        ("chat_sources", "in_reply_to", "VARCHAR(500) DEFAULT ''"),
        ("chat_sources", "author", "VARCHAR(500) DEFAULT ''"),
        ("chat_sources", "last_modified_by", "VARCHAR(500) DEFAULT ''"),
        ("chat_sources", "created_date", "VARCHAR(50) DEFAULT ''"),
        ("chat_sources", "last_modified", "VARCHAR(50) DEFAULT ''"),
        ("chat_history", "is_stopped", "INTEGER DEFAULT 0"),
    ]

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, column, ddl in expected_columns:
            if table not in inspector.get_table_names():
                continue
            existing = {c["name"] for c in inspector.get_columns(table)}
            if column in existing:
                continue
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
                logger.info(f"DB 마이그레이션: {table}.{column} 추가")
            except Exception as e:
                logger.warning(f"컬럼 추가 실패 (skip) — {table}.{column}: {e}")


def reset_globals() -> None:
    """싱글톤 리셋 (테스트용)"""
    global _engine, _SessionLocal, _async_engine, _AsyncSessionLocal
    if _engine:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
    # async 엔진은 async dispose가 필요하지만 테스트 시 이벤트 루프가 없을 수 있어
    # 참조만 끊고 GC에 위임 (PoC 수준에서 충분)
    _async_engine = None
    _AsyncSessionLocal = None


# === 비동기 엔진/세션 (점진 도입용 — 기존 sync 경로와 병행) ===


def get_async_engine(db_url: str | None = None) -> AsyncEngine:
    """비동기 SQLAlchemy 엔진 반환 (싱글톤)

    PoC 시점: 인프라만 제공. 라우터는 기존 sync 경로 유지.
    향후 고동시성 핸들러부터 async로 전환 권장.
    """
    global _async_engine
    if _async_engine is None:
        url = db_url or _ASYNC_DB_URL
        # SQLite 디렉토리 생성
        if url.startswith("sqlite+aiosqlite:///"):
            db_path = Path(url.replace("sqlite+aiosqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        _async_engine = create_async_engine(url, echo=False)
    return _async_engine


def get_async_session_factory(
    engine: AsyncEngine | None = None,
) -> async_sessionmaker[AsyncSession]:
    """비동기 세션 팩토리 반환 (싱글톤)"""
    global _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        _AsyncSessionLocal = async_sessionmaker(
            bind=engine or get_async_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _AsyncSessionLocal


def get_async_session() -> AsyncSession:
    """새 비동기 세션 생성

    사용 예:
        async with get_async_session() as session:
            result = await session.execute(select(...))
            await session.commit()
    """
    factory = get_async_session_factory()
    return factory()
