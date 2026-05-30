"""OCR 모듈 — 엔진 추상화 + 이미지 전처리 + 결과 캐싱

사용:
    from src.parsers.ocr import get_ocr_engine

    engine = get_ocr_engine()  # settings.ocr_engine 기반 자동 선택 + fallback
    text = engine.ocr_image(png_bytes)
"""

from __future__ import annotations

from src.parsers.ocr.base import BaseOcrEngine
from src.parsers.ocr.paddle import PaddleEngine
from src.parsers.ocr.tesseract import TesseractEngine
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 프로세스 단위 싱글톤 — 모델 재로딩 비용 회피
_engine_cache: BaseOcrEngine | None = None


def get_ocr_engine(force: str | None = None) -> BaseOcrEngine:
    """OCR 엔진 인스턴스 반환 (settings.ocr_engine 기반)

    선택 정책:
    - force가 주어지면 해당 엔진 강제 (테스트용)
    - settings.ocr_engine == "tesseract" → TesseractEngine
    - settings.ocr_engine == "paddle" → PaddleEngine (미설치 시 Tesseract fallback)
    - settings.ocr_engine == "auto" → Paddle 우선, 미설치면 Tesseract

    Returns:
        BaseOcrEngine 인스턴스 (싱글톤). 모든 엔진이 미설치면 NoOcrEngine 반환.
    """
    global _engine_cache
    if _engine_cache is not None and force is None:
        return _engine_cache

    choice = (force or settings.ocr_engine or "auto").lower()
    languages = settings.ocr_languages

    engine: BaseOcrEngine
    if choice == "tesseract":
        engine = TesseractEngine(languages=languages)
    elif choice == "paddle":
        candidate = PaddleEngine(languages=languages)
        if candidate.is_available():
            engine = candidate
        else:
            logger.warning("paddle 요청됐으나 미설치 → tesseract fallback")
            engine = TesseractEngine(languages=languages)
    else:  # "auto"
        candidate = PaddleEngine(languages=languages)
        if candidate.is_available():
            engine = candidate
            logger.info("OCR 엔진 자동 선택: paddle")
        else:
            engine = TesseractEngine(languages=languages)
            logger.info("OCR 엔진 자동 선택: tesseract (paddle 미설치)")

    if force is None:
        _engine_cache = engine
    return engine


def reset_engine_cache() -> None:
    """엔진 싱글톤 리셋 (테스트용)"""
    global _engine_cache
    _engine_cache = None


__all__ = [
    "BaseOcrEngine",
    "TesseractEngine",
    "PaddleEngine",
    "get_ocr_engine",
    "reset_engine_cache",
]
