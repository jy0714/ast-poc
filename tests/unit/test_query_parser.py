"""질의 파서 유닛 테스트"""

import pytest

from src.rag.query_parser import ParsedQuery, parse_query


# === 기본 동작 ===


class TestBasic:
    def test_empty_query(self):
        """빈 질의"""
        result = parse_query("")
        assert result.cleaned == ""
        assert result.filters == {}

    def test_simple_query_no_filters(self):
        """필터 없는 단순 질의 — 쿼리 그대로 유지"""
        result = parse_query("비용 관련 내용 알려줘")
        assert result.cleaned == "비용 관련 내용 알려줘"
        assert result.filters == {}
        assert result.intent == "search"

    def test_original_preserved(self):
        """원본 질의 보존"""
        q = "이메일에서 비용 관련 내용 찾아줘"
        result = parse_query(q)
        assert result.original == q


# === 소스 타입 추출 ===


class TestSourceType:
    def test_email_filter(self):
        """이메일 소스 타입 추출"""
        result = parse_query("이메일에서 비용 관련 내용 찾아줘")
        assert result.filters.get("source_type") == "email"

    def test_document_filter(self):
        """문서 소스 타입 추출"""
        result = parse_query("문서 중에서 감사 보고서 찾아줘")
        assert result.filters.get("source_type") == "document"

    def test_teams_chat_filter(self):
        """Teams 채팅 소스 타입 추출"""
        result = parse_query("팀즈 채팅에서 회의 내용 찾아줘")
        assert result.filters.get("source_type") == "teams_chat"

    def test_attachment_filter(self):
        """첨부파일 소스 타입 추출"""
        result = parse_query("첨부파일에서 엑셀 찾아줘")
        assert result.filters.get("source_type") == "attachment"

    def test_source_type_removed_from_query(self):
        """소스 타입 표현이 쿼리에서 제거됨"""
        result = parse_query("이메일에서 비용 관련 내용")
        assert "이메일에서" not in result.cleaned
        assert "비용" in result.cleaned

    def test_no_source_type(self):
        """소스 타입 키워드 없으면 필터 없음"""
        result = parse_query("비용 분석 결과 알려줘")
        assert "source_type" not in result.filters


# === 날짜 추출 ===


class TestDateFilter:
    def test_year_month(self):
        """2025년 3월 → 날짜 범위"""
        result = parse_query("2025년 3월 이메일 찾아줘")
        assert result.filters.get("date_range_start") == "2025-03-01"
        assert result.filters.get("date_range_end") == "2025-04-01"

    def test_year_only(self):
        """2025년 → 전체 연도"""
        result = parse_query("2025년 감사 보고서")
        assert result.filters.get("date_range_start") == "2025-01-01"
        assert result.filters.get("date_range_end") == "2025-12-31"

    def test_december(self):
        """12월 — 경계값"""
        result = parse_query("2025년 12월 내용")
        assert result.filters.get("date_range_start") == "2025-12-01"
        assert result.filters.get("date_range_end") == "2025-12-31"

    def test_month_range(self):
        """1월부터 3월까지"""
        result = parse_query("1월부터 3월까지 이메일 찾아줘")
        assert "date_range_start" in result.filters
        assert "date_range_end" in result.filters

    def test_iso_date(self):
        """ISO 형식 날짜"""
        result = parse_query("2025-01-01부터 2025-03-31까지 내용")
        assert result.filters.get("date_range_start") == "2025-01-01"
        assert result.filters.get("date_range_end") == "2025-03-31"

    def test_no_date(self):
        """날짜 없으면 필터 없음"""
        result = parse_query("비용 관련 내용 알려줘")
        assert "date_range_start" not in result.filters
        assert "date" not in result.filters


# === 참여자 추출 ===


class TestParticipants:
    def test_single_participant(self):
        """단일 참여자"""
        result = parse_query("홍길동이 보낸 이메일 찾아줘")
        assert "홍길동" in result.filters.get("participants", [])

    def test_multiple_participants(self):
        """복수 참여자"""
        result = parse_query("홍길동이 보낸 김철수에게 온 메일")
        participants = result.filters.get("participants", [])
        assert "홍길동" in participants
        assert "김철수" in participants

    def test_no_participants(self):
        """참여자 없음"""
        result = parse_query("비용 관련 내용 알려줘")
        assert "participants" not in result.filters

    def test_no_duplicate_participants(self):
        """중복 참여자 제거"""
        result = parse_query("홍길동이 보낸 홍길동 관련 메일")
        participants = result.filters.get("participants", [])
        assert participants.count("홍길동") == 1


# === 의도 분석 ===


class TestIntent:
    def test_default_search(self):
        """기본 의도 = search"""
        result = parse_query("비용 내용 찾아줘")
        assert result.intent == "search"

    def test_summarize_intent(self):
        """요약 의도"""
        result = parse_query("이메일 내용을 요약해줘")
        assert result.intent == "summarize"

    def test_compare_intent(self):
        """비교 의도"""
        result = parse_query("두 보고서의 차이를 비교해줘")
        assert result.intent == "compare"

    def test_timeline_intent(self):
        """타임라인 의도"""
        result = parse_query("사건 경과를 시간순으로 정리해줘")
        assert result.intent == "timeline"


# === 복합 질의 ===


class TestComplex:
    def test_source_and_date(self):
        """소스 타입 + 날짜 동시 추출"""
        result = parse_query("2025년 3월 이메일에서 비용 관련 내용 찾아줘")
        assert result.filters.get("source_type") == "email"
        assert result.filters.get("date_range_start") == "2025-03-01"
        assert "비용" in result.cleaned

    def test_source_date_participant(self):
        """소스 + 날짜 + 참여자"""
        result = parse_query("홍길동이 보낸 2025년 3월 이메일에서 감사 내용")
        assert result.filters.get("source_type") == "email"
        assert "홍길동" in result.filters.get("participants", [])
        assert "date_range_start" in result.filters

    def test_cleaned_not_empty(self):
        """필터 제거 후 쿼리가 비면 원본 사용"""
        result = parse_query("이메일에서")
        assert result.cleaned  # 빈 문자열이 아님


# === 필터 병합 (engine 내부) ===


class TestMergeFilters:
    def test_merge_import(self):
        """_merge_filters 임포트 가능"""
        from src.rag.engine import _merge_filters

        result = _merge_filters({"source_type": "email"}, {"source_type": "document"})
        assert result["source_type"] == "document"  # 명시 필터 우선

    def test_merge_both_none(self):
        from src.rag.engine import _merge_filters

        assert _merge_filters({}, None) is None

    def test_merge_parsed_only(self):
        from src.rag.engine import _merge_filters

        result = _merge_filters({"source_type": "email"}, None)
        assert result == {"source_type": "email"}

    def test_merge_explicit_only(self):
        from src.rag.engine import _merge_filters

        result = _merge_filters({}, {"source_type": "document"})
        assert result == {"source_type": "document"}
