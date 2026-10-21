"""LLM 라우터 유닛 테스트 — Ollama/OpenAI 없이 모킹"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.router import LLMRouter


@pytest.fixture
def router():
    return LLMRouter()


class TestGetLLM:
    def test_secure_mode_uses_ollama(self, router):
        """보안 모드 ON → Ollama"""
        with patch("src.llm.router.settings") as mock_settings:
            mock_settings.is_secure_mode = True
            mock_settings.ollama_llm_model = "test-model"
            mock_settings.ollama_base_url = "http://localhost:11434"

            # ChatOllama import 모킹
            with patch("src.llm.router.LLMRouter._get_ollama_llm") as mock_get:
                mock_get.return_value = MagicMock()
                router.get_llm(secure_mode=True)
                mock_get.assert_called_once()

    def test_open_mode_uses_openai(self, router):
        """보안 모드 OFF + provider=openai → OpenAI"""
        with patch("src.llm.router.settings") as mock_settings:
            mock_settings.is_secure_mode = False
            mock_settings.external_llm_provider = "openai"
            with patch("src.llm.router.LLMRouter._get_openai_llm") as mock_get:
                mock_get.return_value = MagicMock()
                router.get_llm(secure_mode=False)
                mock_get.assert_called_once()

    def test_open_mode_uses_anthropic(self, router):
        """보안 모드 OFF + provider=anthropic → Anthropic"""
        with patch("src.llm.router.settings") as mock_settings:
            mock_settings.is_secure_mode = False
            mock_settings.external_llm_provider = "anthropic"
            with patch("src.llm.router.LLMRouter._get_anthropic_llm") as mock_get:
                mock_get.return_value = MagicMock()
                router.get_llm(secure_mode=False)
                mock_get.assert_called_once()

    def test_ollama_passes_num_ctx(self, router):
        """로컬 LLM에 LLM_NUM_CTX를 넘긴다 (미지정 시 Ollama가 모델 최대 컨텍스트로 VRAM을 잡음)"""
        with patch("src.llm.router.settings") as mock_settings, \
                patch("langchain_ollama.ChatOllama") as mock_chat:
            mock_settings.ollama_llm_model = "test-model"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.llm_temperature = 0.1
            mock_settings.llm_num_ctx = 16384
            router._get_ollama_llm()
            assert mock_chat.call_args.kwargs["num_ctx"] == 16384

    def test_unknown_provider_raises(self, router):
        """알 수 없는 provider면 ValueError"""
        with patch("src.llm.router.settings") as mock_settings:
            mock_settings.is_secure_mode = False
            mock_settings.external_llm_provider = "bogus"
            with pytest.raises(ValueError, match="external_llm_provider"):
                router.get_llm(secure_mode=False)

    def test_openai_no_key_raises(self, router):
        """OpenAI API 키 없으면 에러"""
        with patch("src.llm.router.settings") as mock_settings:
            mock_settings.openai_api_key = ""
            with pytest.raises(ValueError, match="API 키"):
                router._get_openai_llm()

    def test_anthropic_no_key_raises(self, router):
        """Anthropic API 키 없으면 에러"""
        with patch("src.llm.router.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            with pytest.raises(ValueError, match="API 키"):
                router._get_anthropic_llm()


class TestBuildMessages:
    def test_build_with_context(self, router):
        """컨텍스트 포함 메시지 조합"""
        messages = router._build_messages(
            "비용 관련 내용은?",
            ["이메일 내용 1", "문서 내용 2"],
        )
        assert len(messages) == 2
        assert messages[0][0] == "system"
        assert "감사" in messages[0][1]  # 시스템 프롬프트
        assert messages[1][0] == "human"
        assert "비용 관련 내용은?" in messages[1][1]
        assert "[출처 1]" in messages[1][1]
        assert "[출처 2]" in messages[1][1]

    def test_build_empty_context(self, router):
        """컨텍스트 없음"""
        messages = router._build_messages("질문", [])
        assert "(검색 결과 없음)" in messages[1][1]

    def test_extra_system_warning_appended(self, router):
        """extra_system_warning이 시스템 프롬프트에 삽입됨"""
        warning = "주의: 검색 결과가 질문과 관련 없을 수 있습니다."
        messages = router._build_messages("질문", ["내용"], extra_system_warning=warning)
        assert warning in messages[0][1]
        # 기존 시스템 프롬프트도 유지
        assert "감사" in messages[0][1]

    def test_no_extra_warning_by_default(self, router):
        """extra_system_warning 미전달 시 기존 프롬프트만"""
        messages = router._build_messages("질문", ["내용"])
        assert "## 추가 주의" not in messages[0][1]


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_calls_llm(self, router):
        """generate가 LLM을 호출하는지 확인"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="테스트 답변입니다.")

        with patch.object(router, "get_llm", return_value=mock_llm):
            answer = await router.generate("질문", ["컨텍스트"])

        assert answer == "테스트 답변입니다."
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_invokes_on_complete_with_token_usage(self, router):
        """generate 완료 시 on_complete 콜백으로 토큰 사용량 전달"""
        response = MagicMock(
            content="답변",
            usage_metadata={"input_tokens": 200, "output_tokens": 30, "total_tokens": 230},
            response_metadata={},
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = response

        captured: dict = {}
        with (
            patch.object(router, "get_llm", return_value=mock_llm),
            patch("src.llm.router.log_token_usage"),
            patch("src.llm.router.log_token_usage_detail"),
        ):
            await router.generate(
                "질문", ["컨텍스트"], secure_mode=False,
                case_id="case1", on_complete=captured.update,
            )

        assert captured["input_tokens"] == 200
        assert captured["output_tokens"] == 30
        assert captured["case_id"] == "case1"
        assert captured["input_source"] == "openai_api"

    @pytest.mark.asyncio
    async def test_generate_error_raises(self, router):
        """LLM 호출 실패 시 ConnectionError"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("connection refused")

        with (
            patch.object(router, "get_llm", return_value=mock_llm),
            pytest.raises(ConnectionError, match="응답 생성 실패"),
        ):
            await router.generate("질문", ["컨텍스트"])


class TestGenerateStream:
    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, router):
        """스트리밍이 토큰을 yield하는지 확인"""
        mock_chunks = [
            MagicMock(content="안녕"),
            MagicMock(content="하세요"),
            MagicMock(content=""),
            MagicMock(content="!"),
        ]

        mock_llm = MagicMock()

        async def mock_astream(messages):
            for chunk in mock_chunks:
                yield chunk

        mock_llm.astream = mock_astream

        with patch.object(router, "get_llm", return_value=mock_llm):
            tokens = []
            async for token in router.generate_stream("질문", ["컨텍스트"]):
                tokens.append(token)

        assert tokens == ["안녕", "하세요", "!"]

    @pytest.mark.asyncio
    async def test_stream_logs_token_usage_on_complete(self, router):
        """스트리밍 완료 시 finally에서 on_complete 호출 (추정 경로)"""
        mock_chunks = [MagicMock(content="안녕"), MagicMock(content="하세요")]
        # 마지막 chunk에 usage 메타데이터 없음 → 추정 경로
        for ch in mock_chunks:
            ch.usage_metadata = None
            ch.response_metadata = {}

        mock_llm = MagicMock()

        async def mock_astream(messages):
            for chunk in mock_chunks:
                yield chunk

        mock_llm.astream = mock_astream

        captured: dict = {}
        with (
            patch.object(router, "get_llm", return_value=mock_llm),
            patch("src.llm.router.log_token_usage"),
            patch("src.llm.router.log_token_usage_detail"),
        ):
            async for _ in router.generate_stream(
                "질문", ["컨텍스트"], case_id="c2", on_complete=captured.update
            ):
                pass

        assert captured["case_id"] == "c2"
        assert captured["input_source"] == "estimated"
        assert captured["output_tokens"] > 0  # "안녕하세요" 추정


# === 보안 모드 OFF 외부 API 격리 (v4.4 §6) ===

CASE_TEXT = "케이스기밀문구 송금 4,500만원 박민수"


class _CaptureLLM:
    """외부 LLM 클라이언트 대역 — 받은 메시지를 그대로 기록"""

    def __init__(self) -> None:
        self.sent: list[list[tuple[str, str]]] = []

    async def ainvoke(self, messages):
        self.sent.append(messages)
        return MagicMock(content="외부 답변", usage_metadata=None, response_metadata={})

    async def astream(self, messages):
        self.sent.append(messages)
        for tok in ("외부", " 답변"):
            yield MagicMock(content=tok, usage_metadata=None, response_metadata={})


def _all_text(sent: list[list[tuple[str, str]]]) -> str:
    return "\n".join(content for msgs in sent for _role, content in msgs)


class TestExternalIsolation:
    @pytest.mark.asyncio
    async def test_generate_drops_case_context(self, router):
        llm = _CaptureLLM()
        with patch.object(router, "get_llm", return_value=llm):
            await router.generate("공개 법령 질문", [CASE_TEXT], secure_mode=False,
                                  extra_system_warning=f"경고 {CASE_TEXT}")
        text = _all_text(llm.sent)
        assert CASE_TEXT not in text and "공개 법령 질문" in text
        assert llm.sent[0][0] == ("system", llm.sent[0][0][1]) and "케이스 자료" in llm.sent[0][0][1]
        assert "Context" not in text and "[출처" not in llm.sent[0][1][1]

    @pytest.mark.asyncio
    async def test_generate_stream_drops_case_context(self, router):
        llm = _CaptureLLM()
        with patch.object(router, "get_llm", return_value=llm):
            out = [t async for t in router.generate_stream("공개 질문", [CASE_TEXT, "둘째 " + CASE_TEXT],
                                                            secure_mode=False)]
        assert "".join(out) == "외부 답변"
        assert CASE_TEXT not in _all_text(llm.sent)
        assert [role for role, _c in llm.sent[0]] == ["system", "human"]

    @pytest.mark.asyncio
    async def test_settings_off_also_isolated(self, router, monkeypatch):
        """secure_mode=None 이면 설정값(OFF)을 따라 격리"""
        from src.utils.config import settings

        monkeypatch.setattr(settings, "security_mode", "off")
        llm = _CaptureLLM()
        with patch.object(router, "get_llm", return_value=llm):
            await router.generate("질문", [CASE_TEXT])
        assert CASE_TEXT not in _all_text(llm.sent)

    @pytest.mark.asyncio
    async def test_secure_mode_keeps_context(self, router):
        llm = _CaptureLLM()
        with patch.object(router, "get_llm", return_value=llm):
            await router.generate("질문", [CASE_TEXT], secure_mode=True)
        assert CASE_TEXT in _all_text(llm.sent)
