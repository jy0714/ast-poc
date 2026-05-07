"""AST PoC 로깅 설정

콘솔: Rich 핸들러 (INFO+).
파일: component별 RotatingFileHandler (WARNING+) — 사후 분석을 위해
`error_logs/{component}.log`에 영구 보존.

component는 모듈 이름(`__name__`)의 두 번째 세그먼트로 자동 추출:
- `src.embeddings.embedding_service` → `embeddings.log`
- `src.llm.router`                   → `llm.log`
- `src.vectorstore.vector_store`     → `vectorstore.log`
- `src.indexing.pipeline`            → `indexing.log`
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

# 파일 로그 위치 — 프로젝트 루트의 error_logs/
_ERROR_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "error_logs"
_FILE_MAX_BYTES = 100 * 1024 * 1024  # 100MB
_FILE_BACKUP_COUNT = 5

# 파일 로그 포맷 — 시간/레벨/모듈:라인/메시지로 사후 분석 용이
_FILE_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# component별 파일 핸들러 캐시 (동일 파일에 여러 핸들러가 붙는 것 방지)
_file_handlers: dict[str, RotatingFileHandler] = {}


def _component_from_name(name: str) -> str:
    """모듈 이름에서 component 식별자 추출."""
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "src":
        return parts[1]
    return parts[0] or "root"


def _get_file_handler(component: str) -> RotatingFileHandler:
    """component별 RotatingFileHandler — 싱글톤으로 재사용."""
    if component in _file_handlers:
        return _file_handlers[component]

    _ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _ERROR_LOG_DIR / f"{component}.log"

    handler = RotatingFileHandler(
        log_path,
        maxBytes=_FILE_MAX_BYTES,
        backupCount=_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(logging.WARNING)  # INFO 초과(WARNING/ERROR/CRITICAL)만 파일에 기록
    handler.setFormatter(_FILE_FORMATTER)
    _file_handlers[component] = handler
    return handler


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Rich 콘솔(INFO+) + 파일(WARNING+) 핸들러 로거.

    파일 출력은 `error_logs/{component}.log`에 5MB × 5개 회전으로 보존.
    component는 모듈 경로의 두 번째 세그먼트.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        console_handler = RichHandler(
            level=level,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

        file_handler = _get_file_handler(_component_from_name(name))
        logger.addHandler(file_handler)

        logger.setLevel(level)

    return logger
