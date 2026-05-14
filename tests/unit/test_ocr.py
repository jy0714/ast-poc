"""OCR 모듈 유닛 테스트 — 엔진 추상화 / 캐시 / 전처리

실제 OCR 엔진(tesseract/paddle)이 시스템에 없어도 통과하도록 mock 위주로 검증.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest

from src.parsers.ocr import get_ocr_engine, reset_engine_cache
from src.parsers.ocr.base import BaseOcrEngine
from src.parsers.ocr.paddle import PaddleEngine
from src.parsers.ocr.tesseract import TesseractEngine


def _make_png_bytes(text: str = "hi") -> bytes:
    """간단한 PNG 이미지 바이트 생성 (실제 OCR 호출 안 함, 캐시 키 테스트용)"""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow 미설치")
    img = Image.new("RGB", (50, 20), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def reset_caches(monkeypatch, tmp_path):
    """매 테스트마다 엔진/캐시/디스크 상태 초기화"""
    from src.utils.config import settings as _settings

    # 디스크 캐시는 임시 디렉토리로 격리
    monkeypatch.setattr(_settings, "ocr_cache_dir", str(tmp_path / "ocr_cache"))
    monkeypatch.setattr(_settings, "ocr_cache_enabled", True)

    # 엔진 싱글톤/가용성 캐시 리셋
    reset_engine_cache()
    TesseractEngine._available = None
    TesseractEngine._unavailable_reason = ""
    PaddleEngine._available = None
    PaddleEngine._unavailable_reason = ""
    PaddleEngine._model = None
    yield
    reset_engine_cache()


# === 팩토리 동작 ===


class TestEngineFactory:
    def test_force_tesseract(self):
        engine = get_ocr_engine(force="tesseract")
        assert isinstance(engine, TesseractEngine)
        assert engine.name == "tesseract"

    def test_auto_falls_back_to_tesseract_when_paddle_missing(self, monkeypatch):
        """auto 모드에서 paddle 미설치면 tesseract로 fallback"""
        # paddle 가용성을 강제로 False
        monkeypatch.setattr(PaddleEngine, "_available", False)
        from src.utils.config import settings as _settings

        monkeypatch.setattr(_settings, "ocr_engine", "auto")

        engine = get_ocr_engine()
        assert isinstance(engine, TesseractEngine)

    def test_singleton_caches_engine(self):
        """settings.ocr_engine 변경 안 하면 같은 인스턴스 반환"""
        e1 = get_ocr_engine()
        e2 = get_ocr_engine()
        assert e1 is e2


# === Tesseract 엔진 가용성 ===


class TestTesseractAvailability:
    def test_no_pytesseract_returns_false(self, monkeypatch, caplog):
        """pytesseract import 실패 시 1회 WARNING 후 False"""
        import logging
        import sys

        monkeypatch.setitem(sys.modules, "pytesseract", None)
        engine = TesseractEngine()
        with caplog.at_level(logging.WARNING):
            assert engine.is_available() is False
            # 두 번 호출해도 WARNING 1번
            engine.is_available()
            engine.is_available()
        warns = [r for r in caplog.records
                 if r.levelname == "WARNING" and "Tesseract" in r.message]
        assert len(warns) == 1

    def test_ocr_image_returns_empty_when_unavailable(self, monkeypatch):
        """가용 안 되면 ocr_image는 빈 문자열"""
        monkeypatch.setattr(TesseractEngine, "_available", False)
        engine = TesseractEngine()
        assert engine.ocr_image(_make_png_bytes()) == ""


# === Paddle 엔진 가용성 ===


class TestPaddleAvailability:
    def test_paddle_missing_returns_false(self, monkeypatch, caplog):
        """paddleocr 미설치 시 1회 WARNING 후 False"""
        import logging
        import sys

        monkeypatch.setitem(sys.modules, "paddleocr", None)
        engine = PaddleEngine()
        with caplog.at_level(logging.WARNING):
            assert engine.is_available() is False
        warns = [r for r in caplog.records
                 if r.levelname == "WARNING" and "PaddleOCR" in r.message]
        assert len(warns) == 1

    def test_paddle_supports_parallel_false(self):
        """paddle GPU 단일 컨텍스트라 페이지 병렬 X"""
        engine = PaddleEngine()
        assert engine.supports_parallel() is False


# === OCR 캐시 ===


class TestOcrCache:
    def test_cache_miss_returns_none(self):
        from src.parsers.ocr import cache

        h = cache.hash_image(_make_png_bytes("first"))
        assert cache.get(h) is None

    def test_cache_hit_returns_text(self):
        from src.parsers.ocr import cache

        img_bytes = _make_png_bytes("second")
        h = cache.hash_image(img_bytes)
        cache.put(h, "인식된 텍스트")
        assert cache.get(h) == "인식된 텍스트"

    def test_disabled_cache_always_misses(self, monkeypatch):
        from src.parsers.ocr import cache
        from src.utils.config import settings as _settings

        monkeypatch.setattr(_settings, "ocr_cache_enabled", False)
        h = cache.hash_image(_make_png_bytes("disabled"))
        cache.put(h, "x")  # no-op
        assert cache.get(h) is None

    def test_hash_deterministic(self):
        from src.parsers.ocr import cache

        b = _make_png_bytes("same")
        assert cache.hash_image(b) == cache.hash_image(b)

    def test_hash_differs_for_different_content(self):
        from src.parsers.ocr import cache

        # PIL이 항상 같은 PNG를 만들도록 다른 색의 이미지로
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow 미설치")

        def make(color: str) -> bytes:
            img = Image.new("RGB", (50, 20), color=color)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        h1 = cache.hash_image(make("white"))
        h2 = cache.hash_image(make("black"))
        assert h1 != h2


# === 이미지 전처리 ===


class TestPreprocess:
    def test_preprocess_returns_bytes(self):
        from src.parsers.ocr import preprocess

        result = preprocess.preprocess_image(_make_png_bytes())
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_no_cv2_returns_original(self, monkeypatch):
        """cv2 미설치 시 원본 그대로 반환"""
        import sys

        from src.parsers.ocr import preprocess

        # cv2 캐시 리셋 + sys.modules에서 차단
        preprocess._cv2_available = None
        monkeypatch.setitem(sys.modules, "cv2", None)

        original = _make_png_bytes("orig")
        result = preprocess.preprocess_image(original)
        assert result == original
        # 다시 호출해도 같은 값
        assert preprocess.preprocess_image(original) == original


# === Mock 엔진을 사용한 document_parser 통합 ===


class FakeOcrEngine(BaseOcrEngine):
    """테스트용 mock OCR 엔진 — 항상 정해진 텍스트 반환"""

    name = "fake"

    def __init__(self, text: str = "OCR 결과", available: bool = True):
        super().__init__(languages="kor")
        self._text = text
        self._avail = available
        self.calls = 0

    def is_available(self) -> bool:
        return self._avail

    def ocr_image(self, image_bytes: bytes) -> str:
        self.calls += 1
        return self._text

    def supports_parallel(self) -> bool:
        return True


class TestParserOcrIntegration:
    def test_sparse_pdf_uses_ocr_engine(self, monkeypatch, tmp_path):
        """텍스트 없는 PDF가 mock 엔진으로 OCR 결과 채워짐"""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF 미설치")

        # mock 엔진을 팩토리에 주입
        fake = FakeOcrEngine(text="인식된 본문 텍스트")
        from src.parsers import ocr as ocr_module

        monkeypatch.setattr(ocr_module, "_engine_cache", fake)

        # 빈 페이지 2개짜리 PDF
        path = tmp_path / "scan.pdf"
        doc = fitz.open()
        for _ in range(2):
            doc.new_page()
        doc.save(str(path))
        doc.close()

        from src.parsers.document_parser import DocumentParser

        parser = DocumentParser()
        result = parser.parse(path)

        assert "인식된 본문 텍스트" in result.content
        assert result.metadata["is_ocr"] is True
        assert result.metadata["scan_pdf"] is False  # OCR로 채웠으므로
        assert fake.calls == 2  # 2 페이지 모두 OCR 호출
