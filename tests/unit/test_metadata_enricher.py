"""메타데이터 enricher 유닛 테스트

토픽 추출, 메타데이터 보강, ChromaDB 직렬화/역직렬화 검증.
"""

import pytest

from src.chunkers.chunker import Chunk
from src.chunkers.metadata_enricher import (
    deserialize_metadata_from_chroma,
    enrich_chunks,
    extract_topics,
    serialize_metadata_for_chroma,
)


# === 토픽 추출 테스트 ===


class TestExtractTopics:
    def test_korean_topics(self):
        """한국어 텍스트에서 토픽 추출"""
        text = "감사 보고서에 따르면 비용 분석 결과 감사 항목의 비용이 증가했습니다. 감사 결과를 바탕으로 개선이 필요합니다."
        topics = extract_topics(text, max_topics=3)
        assert len(topics) <= 3
        assert len(topics) > 0
        # "감사"가 가장 빈번하므로 첫 번째 토픽이어야 함
        assert "감사" in topics

    def test_english_topics(self):
        """영어 텍스트에서 토픽 추출"""
        text = "The audit report shows that cost analysis reveals significant cost increases. The audit findings require immediate audit action."
        topics = extract_topics(text, max_topics=3)
        assert len(topics) > 0
        assert "audit" in topics

    def test_mixed_language_topics(self):
        """한영 혼합 텍스트"""
        text = "감사팀 audit report 감사 보고서 검토 audit 결과 분석"
        topics = extract_topics(text)
        assert len(topics) > 0

    def test_empty_text(self):
        """빈 텍스트"""
        assert extract_topics("") == []
        assert extract_topics("   ") == []
        assert extract_topics("짧은") == []

    def test_max_topics_limit(self):
        """최대 토픽 수 제한"""
        text = "감사 보고서 비용 분석 항목 결과 개선 방안 계획 진행 검토 승인" * 3
        topics = extract_topics(text, max_topics=3)
        assert len(topics) <= 3

    def test_stopwords_excluded(self):
        """불용어 제외"""
        text = "이것은 하는 것입니다. 있는 되는 위해 대한 통해 감사 보고서입니다."
        topics = extract_topics(text)
        assert "이것" not in topics
        assert "하는" not in topics


# === 청크 메타데이터 보강 테스트 ===


class TestEnrichChunks:
    def _make_chunks(self, count: int = 3) -> list[Chunk]:
        return [
            Chunk(
                content=f"감사 보고서 내용 {i}번입니다. 비용 분석 결과를 포함합니다." * 3,
                metadata={"filename": f"doc{i}.pdf"},
                source_type="document",
            )
            for i in range(count)
        ]

    def test_case_id_attached(self):
        """case_id가 모든 청크에 부착"""
        chunks = self._make_chunks()
        enrich_chunks(chunks, case_id="CASE_001")
        for chunk in chunks:
            assert chunk.metadata["case_id"] == "CASE_001"

    def test_topics_attached(self):
        """topics가 모든 청크에 부착"""
        chunks = self._make_chunks()
        enrich_chunks(chunks, case_id="CASE_001")
        for chunk in chunks:
            assert "topics" in chunk.metadata
            assert isinstance(chunk.metadata["topics"], list)

    def test_existing_metadata_preserved(self):
        """기존 메타데이터가 유지"""
        chunks = self._make_chunks(1)
        chunks[0].metadata["author"] = "홍길동"
        enrich_chunks(chunks, case_id="C001")
        assert chunks[0].metadata["author"] == "홍길동"
        assert chunks[0].metadata["case_id"] == "C001"

    def test_empty_chunks_list(self):
        """빈 리스트"""
        result = enrich_chunks([], case_id="C001")
        assert result == []


# === ChromaDB 메타데이터 직렬화 테스트 ===


class TestSerializeMetadata:
    def test_primitive_types_unchanged(self):
        """기본 타입은 그대로 유지"""
        meta = {"name": "test", "count": 10, "score": 0.95, "active": True}
        result = serialize_metadata_for_chroma(meta)
        assert result == meta

    def test_list_serialized_to_json(self):
        """list → JSON 문자열"""
        meta = {"participants": ["김감사", "박대리"]}
        result = serialize_metadata_for_chroma(meta)
        assert result["participants"] == '["김감사", "박대리"]'

    def test_dict_serialized_to_json(self):
        """dict → JSON 문자열"""
        meta = {"source": {"email": "test@example.com", "type": "pst"}}
        result = serialize_metadata_for_chroma(meta)
        assert isinstance(result["source"], str)
        assert "test@example.com" in result["source"]

    def test_none_values_excluded(self):
        """None 값 제외"""
        meta = {"name": "test", "empty": None}
        result = serialize_metadata_for_chroma(meta)
        assert "empty" not in result
        assert result["name"] == "test"

    def test_other_types_stringified(self):
        """기타 타입은 str() 변환"""
        from datetime import datetime

        meta = {"date": datetime(2026, 3, 14)}
        result = serialize_metadata_for_chroma(meta)
        assert isinstance(result["date"], str)


class TestDeserializeMetadata:
    def test_json_list_restored(self):
        """JSON 문자열 → list 복원"""
        meta = {"participants": '["김감사", "박대리"]'}
        result = deserialize_metadata_from_chroma(meta)
        assert result["participants"] == ["김감사", "박대리"]

    def test_json_dict_restored(self):
        """JSON 문자열 → dict 복원"""
        meta = {"source": '{"email": "test@example.com"}'}
        result = deserialize_metadata_from_chroma(meta)
        assert result["source"] == {"email": "test@example.com"}

    def test_primitive_types_unchanged(self):
        """기본 타입은 그대로"""
        meta = {"name": "test", "count": 10}
        result = deserialize_metadata_from_chroma(meta)
        assert result == meta

    def test_invalid_json_kept_as_string(self):
        """잘못된 JSON은 문자열 유지"""
        meta = {"data": "[not valid json"}
        result = deserialize_metadata_from_chroma(meta)
        assert result["data"] == "[not valid json"

    def test_roundtrip(self):
        """직렬화 → 역직렬화 라운드트립"""
        original = {
            "name": "test.pdf",
            "participants": ["김감사", "박대리"],
            "topics": ["감사", "비용"],
            "count": 5,
            "active": True,
        }
        serialized = serialize_metadata_for_chroma(original)
        restored = deserialize_metadata_from_chroma(serialized)
        assert restored == original
