"""OCR 결과 디스크 캐시 — 페이지 이미지 sha256 → 인식 텍스트

캐시 키는 (전처리 후) 이미지 바이트의 sha256. 같은 페이지가 여러 케이스에서 반복
인덱싱될 때 OCR 비용을 0으로 만든다. 케이스 격리는 필요 없음 — 같은 이미지면 같은
텍스트이고 메타데이터는 청크 단위에서 별도로 관리됨.

캐시 파일 형식:
    {ocr_cache_dir}/{sha256[:2]}/{sha256}.txt

- 2-character prefix shard로 디렉토리당 파일 수를 제한 (FS 친화)
- 텍스트 파일이라 직접 검사/디버깅 가능
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _cache_path(image_hash: str) -> Path:
    """sha256 → 캐시 파일 경로 (2-character prefix shard)"""
    base = Path(settings.ocr_cache_dir)
    return base / image_hash[:2] / f"{image_hash}.txt"


def hash_image(image_bytes: bytes) -> str:
    """이미지 바이트의 sha256 hex digest"""
    return hashlib.sha256(image_bytes).hexdigest()


def get(image_hash: str) -> str | None:
    """캐시 조회 — hit이면 텍스트, miss/비활성/에러면 None"""
    if not settings.ocr_cache_enabled:
        return None
    path = _cache_path(image_hash)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        logger.debug(f"OCR 캐시 read 실패 ({image_hash[:8]}): {e}")
        return None


def put(image_hash: str, text: str) -> None:
    """캐시 저장 — 빈 텍스트도 저장 (재호출 시 OCR 재시도 방지)"""
    if not settings.ocr_cache_enabled:
        return
    path = _cache_path(image_hash)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # atomic write: temp → rename으로 부분 쓰기 노출 방지
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        logger.debug(f"OCR 캐시 write 실패 ({image_hash[:8]}): {e}")


__all__ = ["hash_image", "get", "put"]
