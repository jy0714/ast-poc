"""메타데이터 보강 — 청크 저장 전 메타데이터 부착 및 직렬화

기능:
- case_id 부착
- 키워드 기반 토픽 추출 (한국어/영어, LLM 의존 없음)
- ChromaDB 호환 메타데이터 직렬화 (list/dict → JSON 문자열)
"""

from __future__ import annotations

import json
from typing import Any

from src.chunkers.chunker import Chunk
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 토픽 추출은 v4 단위 모듈로 이동 (src/units/topics.py)
from src.units.topics import extract_topics  # noqa: E402, F401


def enrich_chunks(
    chunks: list[Chunk],
    case_id: str,
    max_topics: int = 5,
) -> list[Chunk]:
    """청크 리스트에 메타데이터 보강

    - case_id 부착
    - topics 추출 및 부착

    Args:
        chunks: 보강할 청크 리스트
        case_id: 케이스 ID
        max_topics: 청크당 최대 토픽 수

    Returns:
        메타데이터가 보강된 동일 청크 리스트 (in-place 수정)
    """
    for chunk in chunks:
        chunk.metadata["case_id"] = case_id

        # 토픽 추출
        topics = extract_topics(chunk.content, max_topics=max_topics)
        chunk.metadata["topics"] = topics

    logger.info(f"메타데이터 보강 완료: {len(chunks)}개 청크, case_id={case_id}")
    return chunks


def serialize_metadata_for_chroma(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """ChromaDB 호환 메타데이터로 직렬화

    ChromaDB는 메타데이터 값으로 str, int, float, bool만 허용.
    list, dict 등 복합 타입은 JSON 문자열로 변환.

    Args:
        metadata: 원본 메타데이터

    Returns:
        ChromaDB 호환 메타데이터 딕셔너리
    """
    serialized: dict[str, str | int | float | bool] = {}

    for key, value in metadata.items():
        if value is None:
            continue
        elif isinstance(value, (str, int, float, bool)):
            serialized[key] = value
        elif isinstance(value, (list, dict)):
            serialized[key] = json.dumps(value, ensure_ascii=False)
        else:
            serialized[key] = str(value)

    return serialized


def deserialize_metadata_from_chroma(metadata: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB에서 가져온 메타데이터를 원본 형태로 복원

    JSON 문자열로 저장된 list/dict 값을 파싱하여 복원.

    Args:
        metadata: ChromaDB에서 가져온 메타데이터

    Returns:
        복원된 메타데이터 딕셔너리
    """
    deserialized: dict[str, Any] = {}

    for key, value in metadata.items():
        if isinstance(value, str) and value.startswith(("[", "{")):
            try:
                deserialized[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                deserialized[key] = value
        else:
            deserialized[key] = value

    return deserialized
