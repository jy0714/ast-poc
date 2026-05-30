"""EmbeddingService 유닛 테스트

Ollama 서버가 실행 중이어야 통과하는 테스트는 skip 처리.
"""

import pytest

from src.embeddings.embedding_service import (
    KNOWN_DIMENSIONS,
    EmbeddingDimensionError,
    EmbeddingService,
)


def _ollama_model_available() -> bool:
    """Ollama 서버 접근 + 현재 기본 임베딩 모델 존재 여부 확인"""
    try:
        import json
        import urllib.request

        from src.utils.config import settings

        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            model_names = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            return settings.ollama_embed_model in model_names
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_model_available(),
    reason="Ollama 서버가 실행되지 않거나 임베딩 모델이 설치되지 않았습니다",
)


class TestEmbeddingServiceInit:
    def test_default_settings(self):
        """기본 설정값으로 초기화"""
        service = EmbeddingService()
        assert service.model == "bge-m3"
        assert "localhost" in service.base_url

    def test_custom_settings(self):
        """커스텀 설정값"""
        service = EmbeddingService(model="custom-model", base_url="http://custom:11434")
        assert service.model == "custom-model"
        assert service.base_url == "http://custom:11434"

    def test_get_langchain_embeddings_type(self):
        """LangChain 호환 객체 타입 검증"""
        service = EmbeddingService()
        embeddings = service.get_langchain_embeddings()
        from langchain_ollama import OllamaEmbeddings

        assert isinstance(embeddings, OllamaEmbeddings)

    def test_langchain_embeddings_cached(self):
        """LangChain 객체가 캐싱됨"""
        service = EmbeddingService()
        e1 = service.get_langchain_embeddings()
        e2 = service.get_langchain_embeddings()
        assert e1 is e2


class TestDimensionManagement:
    def test_known_model_dimension(self):
        """알려진 모델은 probe 없이 차원 반환"""
        service = EmbeddingService(model="bge-m3")
        assert service.get_dimension() == 1024

    def test_known_model_nomic(self):
        """nomic-embed-text 차원"""
        service = EmbeddingService(model="nomic-embed-text")
        assert service.get_dimension() == 768

    def test_known_model_with_tag(self):
        """태그가 포함된 모델명도 매칭"""
        service = EmbeddingService(model="bge-m3:latest")
        assert service.get_dimension() == 1024

    def test_dimension_cached(self):
        """차원이 캐싱됨"""
        service = EmbeddingService(model="bge-m3")
        _ = service.get_dimension()
        assert service._cached_dimension == 1024
        # 두 번째 호출은 캐시에서
        assert service.get_dimension() == 1024

    def test_validate_dimension_pass(self):
        """차원 일치 시 에러 없음"""
        service = EmbeddingService(model="bge-m3")
        service.validate_dimension(1024)  # 에러 없이 통과

    def test_validate_dimension_mismatch(self):
        """차원 불일치 시 EmbeddingDimensionError"""
        service = EmbeddingService(model="bge-m3")
        with pytest.raises(EmbeddingDimensionError, match="차원 불일치"):
            service.validate_dimension(768)

    def test_validate_dimension_error_message(self):
        """에러 메시지에 모델명과 차원 정보 포함"""
        service = EmbeddingService(model="nomic-embed-text")
        with pytest.raises(EmbeddingDimensionError) as exc_info:
            service.validate_dimension(1024)
        assert "nomic-embed-text" in str(exc_info.value)
        assert "768" in str(exc_info.value)
        assert "1024" in str(exc_info.value)

    def test_known_dimensions_registry(self):
        """KNOWN_DIMENSIONS에 주요 모델 등록"""
        assert "bge-m3" in KNOWN_DIMENSIONS
        assert "nomic-embed-text" in KNOWN_DIMENSIONS
        assert KNOWN_DIMENSIONS["bge-m3"] == 1024
        assert KNOWN_DIMENSIONS["nomic-embed-text"] == 768


@requires_ollama
class TestEmbeddingServiceWithOllama:
    def test_embed_text(self):
        """단일 텍스트 임베딩"""
        service = EmbeddingService()
        vector = service.embed_text("감사 보고서 검토")
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    def test_embed_texts_batch(self):
        """배치 임베딩"""
        service = EmbeddingService()
        texts = ["감사 보고서", "비용 분석", "회의록 검토"]
        vectors = service.embed_texts(texts)
        assert len(vectors) == 3
        assert all(len(v) > 0 for v in vectors)
        # 모든 벡터의 차원이 동일해야 함
        dims = {len(v) for v in vectors}
        assert len(dims) == 1

    def test_embed_texts_empty(self):
        """빈 리스트"""
        service = EmbeddingService()
        assert service.embed_texts([]) == []

    def test_embed_similar_texts(self):
        """유사한 텍스트는 유사한 벡터"""
        import math

        service = EmbeddingService()
        v1 = service.embed_text("감사 보고서 검토")
        v2 = service.embed_text("감사 보고서 리뷰")
        v3 = service.embed_text("오늘 점심 메뉴 추천")

        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na and nb else 0.0

        sim_similar = cosine_sim(v1, v2)
        sim_different = cosine_sim(v1, v3)
        # 유사한 텍스트의 코사인 유사도가 더 높아야 함
        assert sim_similar > sim_different


class TestEmbeddingServiceErrors:
    def test_connection_error(self):
        """연결 불가능한 서버 → ConnectionError"""
        service = EmbeddingService(base_url="http://localhost:19876")
        with pytest.raises(ConnectionError):
            service.embed_text("테스트")


# === 신규: 시간 제한 / 에러 분기 / 분할 깊이 / cancel / 진단 ===


class TestTextStats:
    """진단 헬퍼 — 길이/바이트/유니코드 분포"""

    def test_text_stats_basic(self):
        from src.embeddings.embedding_service import text_stats

        s = text_stats("안녕 abc 漢字")
        assert s["char_count"] == 9
        assert s["byte_count"] > 9  # 한글/한자가 멀티바이트
        assert s["est_tokens"] > 0
        assert s["hangul_pct"] > 0
        assert s["ascii_pct"] > 0
        assert s["cjk_pct"] > 0

    def test_text_stats_empty(self):
        from src.embeddings.embedding_service import text_stats

        s = text_stats("")
        assert s["char_count"] == 0
        assert s["byte_count"] == 0
        assert s["preview"] == ""

    def test_text_stats_preview_capped(self):
        from src.embeddings.embedding_service import text_stats

        long_text = "x" * 1000
        s = text_stats(long_text)
        assert len(s["preview"]) == 200


class TestPrepareTexts:
    """전처리 — sanitize + 빈 텍스트 제외 + 문자/바이트 truncate"""

    def test_byte_truncation_kicks_in(self, monkeypatch):
        """문자 수 한도 미만이지만 바이트 한도 초과 시 truncate"""
        from src.embeddings.embedding_service import EmbeddingService
        from src.utils.config import settings

        # 한자 1자 = 3바이트. 1000자 = 3000바이트지만 byte 한도 1000으로 낮추면 truncate
        monkeypatch.setattr(settings, "embed_max_bytes", 1000)
        service = EmbeddingService()
        text = "漢" * 1000  # 1000자, 3000바이트
        cleaned, skipped, truncated = service._prepare_texts([text])
        assert len(cleaned) == 1
        assert truncated == 1
        # 바이트 한도 안에 들어와야 함
        assert len(cleaned[0][1].encode("utf-8")) <= 1000

    def test_empty_skipped(self):
        from src.embeddings.embedding_service import EmbeddingService

        service = EmbeddingService()
        cleaned, skipped, truncated = service._prepare_texts(["hi", "", "  ", "world"])
        assert len(cleaned) == 2
        assert skipped == 2


class TestSplitIntoBatches:
    """배치 분할 — 짧은/긴 텍스트 분리 + 길이 정렬"""

    def test_short_and_long_separated(self, monkeypatch):
        from src.embeddings.embedding_service import EmbeddingService
        from src.utils.config import settings

        monkeypatch.setattr(settings, "embed_short_text_threshold", 5)
        service = EmbeddingService()
        texts = ["abc", "x", "long text 12345", "another long one"]
        batches = service._split_into_batches(texts, batch_size=10)
        # 짧은 그룹 1개 + 긴 그룹 1개 = 2개 배치
        assert len(batches) == 2
        # 첫 배치는 짧은 텍스트만 (인덱스 0=abc, 1=x)
        first_indices = batches[0][1]
        assert all(len(texts[i]) < 5 for i in first_indices)
        # 두 번째 배치는 긴 텍스트만 (인덱스 2,3)
        second_indices = batches[1][1]
        assert all(len(texts[i]) >= 5 for i in second_indices)

    def test_indices_preserve_original_position(self):
        """정렬 후에도 indices는 원래 위치를 가리켜야 함"""
        from src.embeddings.embedding_service import EmbeddingService

        service = EmbeddingService()
        texts = ["medium text", "short", "very very very long text " * 10]
        batches = service._split_into_batches(texts, batch_size=10)
        # 모든 batch의 indices 합집합이 원본 인덱스 전체를 커버
        all_indices = sorted(idx for _, idxs, _ in batches for idx in idxs)
        assert all_indices == list(range(len(texts)))


class TestEmbedAsyncControl:
    """embed_texts_async 제어 흐름 — Ollama 호출 mock"""

    @pytest.mark.asyncio
    async def test_total_timeout_triggers_failure(self, monkeypatch):
        """전체 시간 제한 초과 시 부분 결과 반환 + 나머지 영구 실패"""
        from src.embeddings.embedding_service import EmbeddingService
        from src.utils.config import settings

        monkeypatch.setattr(settings, "embed_total_timeout_sec", 0)  # 즉시 timeout
        monkeypatch.setattr(settings, "embed_batch_size", 2)

        service = EmbeddingService()
        result = await service.embed_texts_async(["a", "b", "c"])
        # 모든 텍스트가 시간 초과로 실패해야 함
        assert all(v == [] for v in result)
        assert len(service._last_failed_indices) == 3
        for idx in service._last_failed_indices:
            assert service._last_failed_diagnostics[idx]["error_type"] == "total_timeout"

    @pytest.mark.asyncio
    async def test_cancel_event_triggers_failure(self, monkeypatch):
        """cancel_event set 시 즉시 중단"""
        import threading

        from src.embeddings.embedding_service import EmbeddingService
        from src.utils.config import settings

        monkeypatch.setattr(settings, "embed_total_timeout_sec", 60)
        monkeypatch.setattr(settings, "embed_batch_size", 2)

        cancel = threading.Event()
        cancel.set()  # 시작 전에 set → 모든 배치가 cancel로 처리됨

        service = EmbeddingService()
        result = await service.embed_texts_async(
            ["a", "b", "c"], cancel_event=cancel
        )
        assert all(v == [] for v in result)
        for idx in service._last_failed_indices:
            assert service._last_failed_diagnostics[idx]["error_type"] == "cancelled"


class TestRetryPolicy:
    """에러 유형별 재시도 정책 — _embed_batch_with_retry_async를 직접 호출"""

    @pytest.mark.asyncio
    async def test_timeout_immediate_failure_no_wait(self, monkeypatch):
        """timeout은 재시도 없이 즉시 실패 (subdivide로 넘어감)"""
        import time

        import httpx

        from src.embeddings.embedding_service import EmbeddingService

        service = EmbeddingService()

        async def fake_post(*args, **kwargs):
            raise httpx.TimeoutException("simulated timeout")

        async with httpx.AsyncClient() as client:
            monkeypatch.setattr(client, "post", fake_post)
            t_start = time.monotonic()
            with pytest.raises(ConnectionError) as exc:
                await service._embed_batch_with_retry_async(
                    client, "http://x/api/embed", ["text"], 1,
                    deadline=t_start + 60, cancel_event=None,
                )
            elapsed = time.monotonic() - t_start
            # timeout은 재시도 0회 → 1초 미만에 실패해야 함
            assert elapsed < 1.0
            assert exc.value._embed_error_type == "timeout"

    @pytest.mark.asyncio
    async def test_http_500_one_retry_only(self, monkeypatch):
        """5xx는 1회만 재시도 (총 2회 호출, 약 2초)"""
        import time

        import httpx

        from src.embeddings.embedding_service import EmbeddingService

        service = EmbeddingService()
        call_count = {"n": 0}

        async def fake_post(*args, **kwargs):
            call_count["n"] += 1
            req = httpx.Request("POST", "http://x")
            raise httpx.HTTPStatusError(
                "500", request=req,
                response=httpx.Response(500, request=req),
            )

        async with httpx.AsyncClient() as client:
            monkeypatch.setattr(client, "post", fake_post)
            t_start = time.monotonic()
            with pytest.raises(ConnectionError) as exc:
                await service._embed_batch_with_retry_async(
                    client, "http://x/api/embed", ["text"], 1,
                    deadline=t_start + 60, cancel_event=None,
                )
            elapsed = time.monotonic() - t_start
            # 5xx 1회 재시도 + 2초 대기 → 2~3초 사이
            assert call_count["n"] == 2
            assert 1.5 < elapsed < 4.0
            assert exc.value._embed_error_type == "http_500"

    @pytest.mark.asyncio
    async def test_http_4xx_immediate_failure(self, monkeypatch):
        """4xx는 재시도 없이 즉시 실패"""
        import httpx

        from src.embeddings.embedding_service import EmbeddingService

        service = EmbeddingService()
        call_count = {"n": 0}

        async def fake_post(*args, **kwargs):
            call_count["n"] += 1
            req = httpx.Request("POST", "http://x")
            raise httpx.HTTPStatusError(
                "400", request=req,
                response=httpx.Response(400, request=req),
            )

        async with httpx.AsyncClient() as client:
            monkeypatch.setattr(client, "post", fake_post)
            with pytest.raises(ConnectionError) as exc:
                await service._embed_batch_with_retry_async(
                    client, "http://x/api/embed", ["text"], 1,
                    deadline=999999999, cancel_event=None,
                )
            assert call_count["n"] == 1
            assert exc.value._embed_error_type == "http_4xx"


class TestSubdivideDepthLimit:
    """분할 깊이 제한 — embed_max_subdivide_depth 도달 시 영구 실패"""

    @pytest.mark.asyncio
    async def test_depth_limit_marks_permanent_failure(self, monkeypatch):
        """모든 호출이 timeout이면 깊이 N에서 멈추고 모든 청크가 영구 실패로"""
        import httpx

        from src.embeddings.embedding_service import EmbeddingService
        from src.utils.config import settings

        monkeypatch.setattr(settings, "embed_max_subdivide_depth", 2)

        service = EmbeddingService()

        async def always_timeout(*args, **kwargs):
            raise httpx.TimeoutException("always")

        results: list = [None] * 4
        failed: list[int] = []
        diag: dict = {}

        async with httpx.AsyncClient() as client:
            monkeypatch.setattr(client, "post", always_timeout)
            await service._embed_with_subdivide_async(
                client, "http://x/api/embed",
                indices=[0, 1, 2, 3],
                batch=["a", "b", "c", "d"],
                batch_num=1,
                results=results,
                failed_local_indices=failed,
                failure_diag=diag,
                depth=0,
                deadline=99999999,
                cancel_event=None,
            )

        # 4개 모두 실패, 결과는 None
        assert all(r is None for r in results)
        assert sorted(failed) == [0, 1, 2, 3]
        # 진단: depth 도달 또는 timeout으로 분류
        for i in failed:
            assert diag[i]["error_type"] in (
                "subdivide_exhausted", "timeout"
            )
