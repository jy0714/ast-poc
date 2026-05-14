"""OCR 엔진 추상 인터페이스

모든 OCR 엔진은 같은 인터페이스(`ocr_image`)를 통해 호출됨. document_parser는
구체 엔진을 직접 알지 못하고, `get_ocr_engine()` 팩토리에서 받은 인스턴스만 사용한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseOcrEngine(ABC):
    """OCR 엔진 공통 인터페이스

    구현체:
    - TesseractEngine: pytesseract + Tesseract 바이너리
    - PaddleEngine: PaddleOCR (한국어 정확도 우수, GPU/CPU)

    Attributes:
        name: 엔진 식별자 (로그/메트릭용)
        languages: 지원 언어 (tesseract 형식 "eng+kor+chi_sim" 등)
    """

    name: str = "base"

    def __init__(self, languages: str = "eng+kor+chi_sim") -> None:
        self.languages = languages

    @abstractmethod
    def is_available(self) -> bool:
        """엔진이 즉시 사용 가능한지 (의존성/모델/바이너리 확인). 실패 시 1회만 WARNING."""
        ...

    @abstractmethod
    def ocr_image(self, image_bytes: bytes) -> str:
        """이미지(PNG/JPEG bytes)를 OCR하여 텍스트 반환

        Args:
            image_bytes: 인코딩된 이미지 바이트 (PNG 권장)

        Returns:
            인식된 텍스트 (실패/빈 결과는 빈 문자열)
        """
        ...

    def supports_parallel(self) -> bool:
        """페이지 수준 병렬 OCR을 안전하게 받을 수 있는지

        Tesseract는 프로세스 단위 병렬 가능 (CPU 바운드).
        Paddle은 단일 GPU 인스턴스라 직렬 처리가 안전 (False).
        """
        return True
