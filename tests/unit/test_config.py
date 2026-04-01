"""설정 모듈 테스트"""

from src.utils.config import Settings


def test_default_settings():
    """기본 설정값 확인"""
    s = Settings(
        _env_file=None,  # .env 파일 무시
    )
    assert s.ollama_llm_model == "gpt-oss:20b"
    assert s.ollama_embed_model == "bge-m3"
    assert s.security_mode == "on"
    assert s.is_secure_mode is True


def test_security_mode_toggle():
    """보안 모드 토글 테스트"""
    s = Settings(security_mode="on", _env_file=None)
    assert s.is_secure_mode is True

    s = Settings(security_mode="off", _env_file=None)
    assert s.is_secure_mode is False

    s = Settings(security_mode="OFF", _env_file=None)
    assert s.is_secure_mode is False


def test_chunk_settings():
    """청킹 설정 기본값 확인"""
    s = Settings(_env_file=None)
    assert s.chunk_size_docs == 1000
    assert s.chunk_overlap_docs == 200
    assert s.chat_window_minutes == 30
