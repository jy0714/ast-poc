"""메타데이터 보강 — 청크 저장 전 메타데이터 부착 및 직렬화

기능:
- case_id 부착
- 키워드 기반 토픽 추출 (한국어/영어, LLM 의존 없음)
- ChromaDB 호환 메타데이터 직렬화 (list/dict → JSON 문자열)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from src.chunkers.chunker import Chunk
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 한국어 불용어 (토픽 추출 시 제외)
_KO_STOPWORDS = frozenset({
    "있다", "하다", "되다", "이다", "것", "수", "등", "및", "또는", "그리고",
    "하는", "있는", "되는", "위해", "대한", "통해", "에서", "으로", "에서는",
    "합니다", "있습니다", "됩니다", "입니다", "것입니다", "바랍니다",
    "이", "그", "저", "이것", "그것", "여기", "거기", "무엇", "어떤",
    "않", "못", "안", "더", "매우", "아주", "정말", "너무",
})

# 영어 불용어
_EN_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "need",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "and", "or", "but", "if", "then", "so", "because", "as", "of",
    "in", "on", "at", "to", "for", "with", "by", "from", "about",
    "not", "no", "nor", "very", "too", "also", "just", "only",
})

# 한국어 단어 패턴 (2글자 이상 한글 단어)
_KO_WORD_RE = re.compile(r"[가-힣]{2,}")
# 영어 단어 패턴 (3글자 이상)
_EN_WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def extract_topics(text: str, max_topics: int = 5) -> list[str]:
    """텍스트에서 핵심 키워드(토픽)를 추출

    빈도 기반으로 상위 키워드를 반환. LLM 없이 경량 처리.

    Args:
        text: 분석할 텍스트
        max_topics: 최대 토픽 수

    Returns:
        토픽 키워드 리스트 (빈도순)
    """
    if not text or len(text.strip()) < 10:
        return []

    words: list[str] = []

    # 한국어 단어 추출
    ko_words = _KO_WORD_RE.findall(text)
    words.extend(w for w in ko_words if w not in _KO_STOPWORDS)

    # 영어 단어 추출 (소문자로 정규화)
    en_words = _EN_WORD_RE.findall(text)
    words.extend(w.lower() for w in en_words if w.lower() not in _EN_STOPWORDS)

    if not words:
        return []

    # 빈도 기반 상위 키워드
    counter = Counter(words)
    return [word for word, _ in counter.most_common(max_topics)]


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
