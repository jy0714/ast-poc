"""AST PoC 로깅 설정"""

import logging
import sys

from rich.logging import RichHandler


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Rich 핸들러를 사용하는 로거 생성"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = RichHandler(
            level=level,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger
