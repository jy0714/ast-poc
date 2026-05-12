"""SQLAlchemy ORM 모델 정의

테이블:
    cases          — 케이스 메타데이터 (이름, 상태, 경로 등)
    indexing_logs   — 인덱싱 실행 이력
    chat_history    — 질의/응답 대화 로그
    chat_sources    — 응답에 사용된 출처 청크 (chat_history FK)
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy 베이스 클래스"""


class CaseModel(Base):
    """케이스 메타데이터 테이블"""

    __tablename__ = "cases"

    case_id: Mapped[str] = mapped_column(String(12), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
    pst_paths: Mapped[str] = mapped_column(Text, default="[]")  # JSON 직렬화
    doc_paths: Mapped[str] = mapped_column(Text, default="[]")  # JSON 직렬화
    total_documents: Mapped[int] = mapped_column(Integer, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, default="")

    # Relationships
    indexing_logs: Mapped[list[IndexingLogModel]] = relationship(back_populates="case", cascade="all, delete-orphan")
    chat_histories: Mapped[list[ChatHistoryModel]] = relationship(back_populates="case", cascade="all, delete-orphan")


class IndexingLogModel(Base):
    """인덱싱 실행 이력 테이블"""

    __tablename__ = "indexing_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String(12), ForeignKey("cases.case_id", ondelete="CASCADE"))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="running")  # running, completed, error, cancelled
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    processed_files: Mapped[int] = mapped_column(Integer, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[str] = mapped_column(Text, default="[]")  # JSON 직렬화
    elapsed_seconds: Mapped[int] = mapped_column(Integer, default=0)

    # Relationship
    case: Mapped[CaseModel] = relationship(back_populates="indexing_logs")


class ChatHistoryModel(Base):
    """채팅 히스토리 테이블"""

    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String(12), ForeignKey("cases.case_id", ondelete="CASCADE"))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, default="")
    security_mode: Mapped[int] = mapped_column(Integer, default=1)  # 1=on, 0=off
    filters: Mapped[str] = mapped_column(Text, default="{}")  # JSON 직렬화
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    # Relationships
    case: Mapped[CaseModel] = relationship(back_populates="chat_histories")
    sources: Mapped[list[ChatSourceModel]] = relationship(back_populates="chat", cascade="all, delete-orphan")


class ChatSourceModel(Base):
    """채팅 응답 출처 테이블"""

    __tablename__ = "chat_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(Integer, ForeignKey("chat_history.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text, default="")
    source_type: Mapped[str] = mapped_column(String(20), default="")
    filename: Mapped[str] = mapped_column(String(500), default="")
    date: Mapped[str] = mapped_column(String(50), default="")
    participants: Mapped[str] = mapped_column(Text, default="[]")  # JSON 직렬화
    subject: Mapped[str] = mapped_column(String(500), default="")
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)
    search_method: Mapped[str] = mapped_column(String(20), default="")
    # 이메일 전용 (다른 source_type에서는 빈 값/리스트). 기존 DB는 init_db의
    # 자동 마이그레이션이 ALTER TABLE로 추가.
    sender: Mapped[str] = mapped_column(String(500), default="")
    recipients: Mapped[str] = mapped_column(Text, default="[]")  # JSON 직렬화
    cc: Mapped[str] = mapped_column(Text, default="[]")  # JSON 직렬화
    attachments: Mapped[str] = mapped_column(Text, default="[]")  # JSON 직렬화
    message_id: Mapped[str] = mapped_column(String(500), default="")
    in_reply_to: Mapped[str] = mapped_column(String(500), default="")
    # Office/PDF 작성자·수정자 추적
    author: Mapped[str] = mapped_column(String(500), default="")
    last_modified_by: Mapped[str] = mapped_column(String(500), default="")
    created_date: Mapped[str] = mapped_column(String(50), default="")
    last_modified: Mapped[str] = mapped_column(String(50), default="")

    # Relationship
    chat: Mapped[ChatHistoryModel] = relationship(back_populates="sources")
