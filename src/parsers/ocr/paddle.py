"""PaddleOCR 엔진 — 한국어/영어/한자 정확도 우수, GPU 활용 가능

paddleocr + paddlepaddle은 무거운 의존성(~4GB)이라 lazy import.
미설치 시 is_available()이 False → 호출자가 다른 엔진으로 fallback.
GPU/CPU는 paddle 자체가 자동 선택 (CUDA 가용 시 GPU).
"""

from __future__ import annotations

import io
import threading

from src.parsers.ocr.base import BaseOcrEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)

# tesseract 언어코드 → paddleocr 언어 코드 매핑.
# paddleocr는 한 인스턴스가 한 언어만 처리 → 가장 첫 언어를 우선 적용.
# 다국어 혼재 문서는 "korean" 모델이 영문도 어느 정도 처리하므로 무난.
_LANG_MAP: dict[str, str] = {
    "kor": "korean",
    "eng": "en",
    "chi_sim": "ch",
    "chi_tra": "chinese_cht",
    "jpn": "japan",
}


class PaddleEngine(BaseOcrEngine):
    """PaddleOCR 기반 엔진 — 한국어 95%+ 정확도, GPU/CPU 자동 선택

    동일 프로세스에서 paddle 모델은 1회만 로드되어 인스턴스에 보존됨.
    GPU 인스턴스는 단일 컨텍스트라 페이지 병렬은 비효율 → supports_parallel=False.
    """

    name = "paddle"

    _import_attempted: bool = False
    _available: bool | None = None
    _unavailable_reason: str = ""
    # paddle 모델은 무거우니 클래스 단위로 1회만 로드 + 동시 호출 보호
    _model = None
    _model_lock = threading.Lock()

    def is_available(self) -> bool:
        if PaddleEngine._available is not None:
            return PaddleEngine._available
        try:
            # paddleocr import는 paddlepaddle도 함께 로드해서 무거우므로 첫 호출에 시간 소요
            from paddleocr import PaddleOCR  # noqa: F401
            PaddleEngine._available = True
            logger.info("PaddleOCR 가용 — 첫 호출 시 모델 로드 (~수 초)")
            return True
        except ImportError as e:
            PaddleEngine._available = False
            PaddleEngine._unavailable_reason = (
                f"paddleocr/paddlepaddle 미설치: {e}. "
                "pip install paddleocr paddlepaddle (또는 paddlepaddle-gpu)"
            )
            logger.warning(
                f"PaddleOCR 비활성화 — {PaddleEngine._unavailable_reason}"
            )
            return False

    def _resolve_lang(self) -> str:
        """tesseract languages 문자열에서 paddle 언어 1개 선택"""
        for tess in self.languages.split("+"):
            tess = tess.strip().lower()
            if tess in _LANG_MAP:
                return _LANG_MAP[tess]
        return "korean"  # 한국 환경 기본

    def _get_model(self):  # noqa: ANN202
        """paddle 모델 인스턴스 (lazy 로드, thread-safe)"""
        if PaddleEngine._model is not None:
            return PaddleEngine._model
        with PaddleEngine._model_lock:
            if PaddleEngine._model is None:
                from paddleocr import PaddleOCR

                lang = self._resolve_lang()
                # use_angle_cls=True: 글자 회전 자동 보정 (스캔 PDF 회전 페이지 대응)
                # show_log=False: paddle 자체 로깅 억제
                PaddleEngine._model = PaddleOCR(
                    use_angle_cls=True, lang=lang, show_log=False,
                )
                logger.info(f"PaddleOCR 모델 로드 완료 (lang={lang})")
        return PaddleEngine._model

    def ocr_image(self, image_bytes: bytes) -> str:
        if not self.is_available():
            return ""
        try:
            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            arr = np.array(img)

            model = self._get_model()
            # paddle ocr는 GPU 단일 컨텍스트 → lock으로 직렬화
            with PaddleEngine._model_lock:
                result = model.ocr(arr, cls=True)

            # paddle 결과 형식: [[[box, (text, conf)], ...]] (페이지 1개)
            if not result or not result[0]:
                return ""
            lines: list[str] = []
            for line in result[0]:
                if not line or len(line) < 2:
                    continue
                _box, text_conf = line[0], line[1]
                if isinstance(text_conf, (tuple, list)) and text_conf:
                    text = text_conf[0]
                    if text and isinstance(text, str):
                        lines.append(text.strip())
            return "\n".join(lines).strip()
        except Exception as e:
            logger.debug(f"PaddleOCR 실패 (page skip): {e}")
            return ""

    def supports_parallel(self) -> bool:
        # GPU 단일 인스턴스 → 페이지 병렬은 컨텍스트 경합. 직렬 처리.
        return False
