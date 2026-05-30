"""OCR 정확도 향상용 이미지 전처리 — deskew, denoise, contrast

OpenCV(opencv-python-headless) 사용. import 실패 시 원본 이미지를 그대로 반환
(전처리 없는 OCR도 문제없이 동작).

전처리 순서:
1. RGB → grayscale
2. denoise (작은 노이즈 제거)
3. deskew (페이지 기울기 보정, 스캔/촬영 PDF에서 흔함)
4. contrast 보정 (CLAHE)

각 단계는 best-effort — 실패 시 이전 단계 결과 유지.
"""

from __future__ import annotations

import io

from src.utils.logger import get_logger

logger = get_logger(__name__)

# OpenCV 가용성 (1회 체크)
_cv2_available: bool | None = None


def _check_cv2() -> bool:
    """cv2 import 가능 여부 (1회 체크 + 캐시)"""
    global _cv2_available
    if _cv2_available is not None:
        return _cv2_available
    try:
        import cv2  # noqa: F401
        _cv2_available = True
        return True
    except ImportError as e:
        _cv2_available = False
        logger.warning(
            f"OpenCV 미설치 — OCR 이미지 전처리 비활성화: {e}. "
            "정확도 향상을 위해 pip install opencv-python-headless"
        )
        return False


def preprocess_image(image_bytes: bytes) -> bytes:
    """OCR 입력 이미지 전처리 → PNG bytes 반환

    cv2 미설치 시 원본을 그대로 반환. 각 단계는 best-effort — 실패하면 이전 단계의
    결과를 다음 단계로 넘김.

    Args:
        image_bytes: 원본 이미지 (PNG/JPEG bytes)

    Returns:
        전처리된 PNG bytes (또는 실패 시 원본)
    """
    if not _check_cv2():
        return image_bytes

    try:
        import cv2
        import numpy as np

        # 디코드
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes

        # 1. grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. denoise (h=10: 적당한 노이즈 제거, 텍스트 보존)
        try:
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
        except Exception:
            denoised = gray

        # 3. deskew — 텍스트 영역 contour의 minAreaRect 기울기 추정
        try:
            angle = _estimate_skew_angle(denoised)
            if abs(angle) > 0.5:  # 0.5도 이상 기울었을 때만 회전 (불필요한 보간 방지)
                deskewed = _rotate(denoised, angle)
            else:
                deskewed = denoised
        except Exception as e:
            logger.debug(f"deskew 실패 (skip): {e}")
            deskewed = denoised

        # 4. CLAHE 대비 보정
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(deskewed)
        except Exception:
            enhanced = deskewed

        # 인코드
        success, encoded = cv2.imencode(".png", enhanced)
        if not success:
            return image_bytes
        return encoded.tobytes()

    except Exception as e:
        logger.debug(f"이미지 전처리 실패 (원본 사용): {e}")
        return image_bytes


def _estimate_skew_angle(gray) -> float:  # noqa: ANN001
    """텍스트의 평균 회전각 추정 (도 단위, -45~+45)

    이진화 후 흑색 픽셀(텍스트)의 minAreaRect 각도를 사용. 빈 페이지나 정렬된
    페이지는 0에 가까운 값을 반환.
    """
    import cv2
    import numpy as np

    # 이진화 (텍스트는 0, 배경은 255)
    _thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:  # 텍스트가 거의 없으면 회전 추정 의미 없음
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV의 minAreaRect는 -90~0 범위로 반환 → 정규화
    if angle < -45:
        angle = 90 + angle
    return float(angle)


def _rotate(gray, angle: float):  # noqa: ANN001, ANN201
    """이미지를 angle도 회전 (배경은 흰색으로 패딩)"""
    import cv2

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


__all__ = ["preprocess_image"]
