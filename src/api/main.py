"""AST PoC - FastAPI 메인 애플리케이션"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import cases, chat, documents, health, indexing, settings as settings_routes
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행"""
    # DB 초기화 (테이블 자동 생성)
    from src.db.database import init_db
    init_db()

    logger.info("[bold green]AST PoC 서버 시작[/]")
    logger.info(f"보안 모드: {'ON (로컬 전용)' if settings.is_secure_mode else 'OFF (외부 API 허용)'}")
    logger.info(f"Ollama LLM: {settings.ollama_llm_model}")
    logger.info(f"Ollama Embed: {settings.ollama_embed_model}")
    yield
    logger.info("[bold red]AST PoC 서버 종료[/]")


app = FastAPI(
    title="AST (Audit Support Tool) PoC",
    description="내부 문서 및 커뮤니케이션 통합 검색 RAG 시스템",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Admin API ===
app.include_router(health.router, tags=["Health"])
app.include_router(cases.router, prefix="/api/admin/cases", tags=["Admin - Cases"])
app.include_router(indexing.router, prefix="/api/admin/indexing", tags=["Admin - Indexing"])
app.include_router(documents.router, prefix="/api/admin/documents", tags=["Admin - Documents"])
app.include_router(settings_routes.router, prefix="/api/admin/settings", tags=["Admin - Settings"])

# === Analyst API ===
app.include_router(chat.router, prefix="/api/analyst/chat", tags=["Analyst - Chat"])
