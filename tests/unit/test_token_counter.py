"""토큰 카운터 + 로깅 유닛 테스트"""

from types import SimpleNamespace

import pytest

from src.utils import token_counter as tc
from src.utils.token_counter import (
    TokenCounter,
    build_usage_record,
    log_token_usage,
    log_token_usage_detail,
)


@pytest.fixture
def counter_no_tiktoken(monkeypatch):
    """tiktoken을 강제로 비활성화한 TokenCounter (문자 기반 추정 경로 테스트)"""
    c = TokenCounter()
    monkeypatch.setattr(c, "_get_encoder", lambda: None)
    return c


class TestEstimateTokens:
    def test_empty_text_is_zero(self, counter_no_tiktoken):
        assert counter_no_tiktoken.estimate_tokens("") == 0

    def test_fallback_formula(self, counter_no_tiktoken):
        """max(utf-8 바이트/3, 문자 수/2) 근사식"""
        text = "abcd"  # 4 bytes, 4 chars → max(4/3, 4/2) = 2
        assert counter_no_tiktoken.estimate_tokens(text) == 2

    def test_korean_uses_byte_path(self, counter_no_tiktoken):
        """한글은 문자당 3바이트라 바이트 경로가 우세"""
        text = "안녕"  # 6 bytes, 2 chars → max(6/3, 2/2) = 2
        assert counter_no_tiktoken.estimate_tokens(text) == 2

    def test_tiktoken_used_when_available(self, monkeypatch):
        """인코더가 있으면 인코더 결과 사용"""
        c = TokenCounter()
        fake_encoder = SimpleNamespace(encode=lambda t: [1, 2, 3, 4, 5])
        monkeypatch.setattr(c, "_get_encoder", lambda: fake_encoder)
        assert c.estimate_tokens("아무 텍스트") == 5


class TestExtractUsage:
    def test_none_response(self):
        assert TokenCounter().extract_usage_from_response(None) is None

    def test_ollama_metadata(self):
        resp = SimpleNamespace(
            response_metadata={"prompt_eval_count": 120, "eval_count": 45},
            usage_metadata=None,
        )
        usage = TokenCounter().extract_usage_from_response(resp)
        assert usage == {"input_tokens": 120, "output_tokens": 45, "source": "ollama_api"}

    def test_openai_usage_metadata(self):
        resp = SimpleNamespace(
            usage_metadata={"input_tokens": 300, "output_tokens": 80, "total_tokens": 380},
            response_metadata={},
        )
        usage = TokenCounter().extract_usage_from_response(resp)
        assert usage == {"input_tokens": 300, "output_tokens": 80, "source": "openai_api"}

    def test_openai_legacy_token_usage(self):
        resp = SimpleNamespace(
            usage_metadata=None,
            response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 7}},
        )
        usage = TokenCounter().extract_usage_from_response(resp)
        assert usage == {"input_tokens": 10, "output_tokens": 7, "source": "openai_api"}

    def test_no_metadata_returns_none(self):
        resp = SimpleNamespace(usage_metadata=None, response_metadata={})
        assert TokenCounter().extract_usage_from_response(resp) is None


class TestBuildUsageRecord:
    def test_uses_actual_usage_when_present(self):
        tsv, detail = build_usage_record(
            question="질문",
            context=["컨텍스트 청크 하나"],
            output_text="답변입니다.",
            model="gemma4:e4b",
            secure_mode=True,
            latency_sec=1.234,
            usage={"input_tokens": 500, "output_tokens": 120, "source": "ollama_api"},
            system_prompt="시스템 프롬프트",
        )
        assert tsv["input_tokens"] == 500
        assert tsv["output_tokens"] == 120
        assert tsv["total_tokens"] == 620
        assert tsv["input_source"] == "ollama_api"
        assert tsv["security_mode"] == "on"
        assert tsv["context_chunks"] == 1
        assert detail["breakdown"]["total_input_tokens"] > 0

    def test_falls_back_to_estimate(self):
        tsv, _ = build_usage_record(
            question="질문",
            context=["컨텍스트"],
            output_text="답변",
            model="o3",
            secure_mode=False,
            latency_sec=0.5,
            usage=None,
            system_prompt="시스템",
        )
        assert tsv["input_source"] == "estimated"
        assert tsv["input_tokens"] > 0
        assert tsv["security_mode"] == "off"


class TestLogging:
    def test_tsv_written_with_header(self, tmp_path, monkeypatch):
        log_file = tmp_path / "token_usage.txt"
        monkeypatch.setattr(tc.settings, "token_log_enabled", True)
        monkeypatch.setattr(tc.settings, "token_log_file", str(log_file))

        record = {
            "timestamp": "2026-05-24 10:00:00",
            "case_id": "abc",
            "question": "질문",
            "model": "gemma4:e4b",
            "security_mode": "on",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_source": "estimated",
            "context_chunks": 3,
            "latency_sec": 1.2,
        }
        log_token_usage(record)
        log_token_usage(record)

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert lines[0].startswith("timestamp\tcase_id\tquestion")  # 헤더
        assert len(lines) == 3  # 헤더 1 + 데이터 2
        assert "\tabc\t" in lines[1]

    def test_disabled_writes_nothing(self, tmp_path, monkeypatch):
        log_file = tmp_path / "token_usage.txt"
        monkeypatch.setattr(tc.settings, "token_log_enabled", False)
        monkeypatch.setattr(tc.settings, "token_log_file", str(log_file))
        log_token_usage({"case_id": "x"})
        assert not log_file.exists()

    def test_detail_respects_flag(self, tmp_path, monkeypatch):
        detail_file = tmp_path / "detail.jsonl"
        monkeypatch.setattr(tc.settings, "token_log_detail", False)
        monkeypatch.setattr(tc.settings, "token_log_detail_file", str(detail_file))
        log_token_usage_detail({"case_id": "x"})
        assert not detail_file.exists()

        monkeypatch.setattr(tc.settings, "token_log_detail", True)
        log_token_usage_detail({"case_id": "x", "output_tokens": 5})
        assert detail_file.exists()
        assert '"case_id": "x"' in detail_file.read_text(encoding="utf-8")

    def test_logging_failure_is_swallowed(self, monkeypatch):
        """파일 경로가 잘못돼도 예외가 전파되지 않아야 함"""
        monkeypatch.setattr(tc.settings, "token_log_enabled", True)
        # 디렉토리로 사용할 수 없는 경로 — 쓰기 실패 유도
        monkeypatch.setattr(tc, "_resolve_path", lambda _: (_ for _ in ()).throw(OSError("boom")))
        # 예외 없이 반환되면 통과
        log_token_usage({"case_id": "x"})
