"""Tesseract OCR 엔진 — pytesseract + 시스템 tesseract 바이너리

가용성은 클래스 변수로 1회만 체크 (반복 ImportError WARNING 방지).
이전 DocumentParser._ocr_page 로직을 이쪽으로 이전.
"""

from __future__ import annotations

import io

from src.parsers.ocr.base import BaseOcrEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TesseractEngine(BaseOcrEngine):
    """Tesseract OCR 엔진

    한국어 정확도 ~70~85% (낮은 DPI, 손글씨, 표에서 떨어짐).
    가벼운 의존성, CPU 바운드, 멀티프로세스 가능.
    """

    name = "tesseract"

    # 가용성 체크 캐시 (프로세스 단위 1회)
    _available: bool | None = None
    _unavailable_reason: str = ""

    def is_available(self) -> bool:
        if TesseractEngine._available is not None:
            return TesseractEngine._available
        try:
            import pytesseract
            from PIL import Image  # noqa: F401
            try:
                pytesseract.get_tesseract_version()
            except Exception as bin_err:
                TesseractEngine._available = False
                TesseractEngine._unavailable_reason = f"tesseract 바이너리 미설치: {bin_err}"
                logger.warning(
                    f"Tesseract OCR 비활성화 — {TesseractEngine._unavailable_reason}"
                )
                return False
            TesseractEngine._available = True
            logger.info("Tesseract OCR 가용 — pytesseract + 바이너리 확인됨")
            return True
        except ImportError as e:
            TesseractEngine._available = False
            TesseractEngine._unavailable_reason = f"pytesseract/Pillow 미설치: {e}"
            logger.warning(
                f"Tesseract OCR 비활성화 — {TesseractEngine._unavailable_reason}"
            )
            return False

    def ocr_image(self, image_bytes: bytes) -> str:
        if not self.is_available():
            return ""
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(img, lang=self.languages)
            return text.strip()
        except Exception as e:
            logger.debug(f"Tesseract OCR 실패 (page skip): {e}")
            return ""

    def supports_parallel(self) -> bool:
        # pytesseract는 내부적으로 tesseract 바이너리를 subprocess로 호출 → 멀티스레드 안전
        return True
