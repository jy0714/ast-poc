"""임베딩 서비스 — Ollama 로컬 임베딩 (항상 로컬)

보안 정책: 임베딩은 보안 모드와 무관하게 항상 로컬 Ollama에서 수행.
외부 API로 임베딩을 전송하지 않음.

모델별 벡터 차원:
- nomic-embed-text: 768
- bge-m3: 1024
- mxbai-embed-large: 1024

모델 변경 시 기존 컬렉션과 벡터 차원 불일치가 발생할 수 있으므로
validate_dimension()으로 호환성을 검증해야 함.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
import threading
import time
import unicodedata

import httpx

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# HTTP 요청 timeout — 빨리 실패하고 작은 배치로 재시도하는 편이 전체 처리량에 유리
_REQUEST_TIMEOUT = 60.0

# ConnectError 재시도 정책 (네트워크 일시 단절). 다른 에러는 별도 정책.
_CONNECT_MAX_RETRIES = 3
_CONNECT_RETRY_BACKOFF_BASE = 2.0  # 2 → 4 → 8

# 5xx 에러 정책: 1회만 짧게 재시도. 같은 배치 같은 입력에 대한 재시도는 의미 없으므로 빠르게 분할로 넘김.
_SERVER_ERROR_RETRY_WAIT = 2.0

# bge-m3 컨텍스트 한도(`/api/ps` 응답상 4096 토큰)에 맞춘 보수적 절단.
# 한글 평균 1자 ≈ 1토큰, 영문 1자 ≈ 0.3토큰. 헤더/마진 고려 2500자에서 절단.
_MAX_CHARS_PER_TEXT = 2500

# 배치 실패 시 binary subdivide 최소 단위 — 이 이하로는 더 못 쪼갬
_MIN_SUBDIVIDE_SIZE = 1


class EmbedTotalTimeoutError(Exception):
    """전체 시간 제한 초과 — 호출자가 부분 결과만 반환하도록 트리거"""


class EmbedCancelledError(Exception):
    """cancel_event가 set되어 임베딩 중단됨"""


# === 진단 헬퍼 ===

def text_stats(text: str) -> dict[str, object]:
    """독성 청크 진단용 통계 (길이, 바이트, 추정 토큰, 유니코드 카테고리 분포)

    bge-m3 토큰 추정: 한글 ≈ 1자/1토큰, ASCII ≈ 1자/0.3토큰.
    Returns:
        dict with: char_count, byte_count, est_tokens, hangul_pct, ascii_pct,
                   cjk_pct, other_pct, preview (200자)
    """
    chars = len(text)
    if chars == 0:
        return {
            "char_count": 0, "byte_count": 0, "est_tokens": 0,
            "hangul_pct": 0, "ascii_pct": 0, "cjk_pct": 0, "other_pct": 0,
            "preview": "",
        }
    bytes_count = len(text.encode("utf-8", errors="ignore"))
    hangul = sum(1 for c in text if "가" <= c <= "힯")
    ascii_n = sum(1 for c in text if ord(c) < 128)
    # CJK 한자 (간체/번체/일본 한자)
    cjk = sum(
        1 for c in text
        if "一" <= c <= "鿿" or "㐀" <= c <= "䶿"
        or "぀" <= c <= "ヿ"  # 히라가나/가타카나
    )
    other = chars - hangul - ascii_n - cjk
    est_tokens = int(hangul * 1.0 + ascii_n * 0.3 + cjk * 1.5 + other * 1.0)
    return {
        "char_count": chars,
        "byte_count": bytes_count,
        "est_tokens": est_tokens,
        "hangul_pct": round(hangul * 100 / chars),
        "ascii_pct": round(ascii_n * 100 / chars),
        "cjk_pct": round(cjk * 100 / chars),
        "other_pct": round(other * 100 / chars),
        "preview": text[:200],
    }


def _format_text_stats(stats: dict[str, object]) -> str:
    """text_stats 결과를 한 줄 로그 메시지로"""
    return (
        f"길이={stats['char_count']}자/{stats['byte_count']}B/~{stats['est_tokens']}tok, "
        f"한글={stats['hangul_pct']}% ASCII={stats['ascii_pct']}% "
        f"CJK={stats['cjk_pct']}% 기타={stats['other_pct']}%"
    )

# 모델별 알려진 벡터 차원 (Ollama 기준)
KNOWN_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 768,
    "bge-m3": 1024,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}


class EmbeddingDimensionError(Exception):
    """임베딩 벡터 차원 불일치 에러"""


class EmbeddingService:
    """Ollama 로컬 임베딩 모델 연동

    사용법:
        service = EmbeddingService()
        vector = service.embed_text("검색할 텍스트")
        vectors = service.embed_texts(["텍스트1", "텍스트2"])

        # 벡터 차원 확인
        dim = service.get_dimension()

        # 컬렉션 호환성 검증
        service.validate_dimension(existing_dim=768)
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        """EmbeddingService 초기화

        Args:
            model: Ollama 임베딩 모델명 (기본: settings.ollama_embed_model)
            base_url: Ollama 서버 URL (기본: settings.ollama_base_url)
        """
        self.model = model or settings.ollama_embed_model
        self.base_url = base_url or settings.ollama_base_url
        self._langchain_embeddings = None
        self._cached_dimension: int | None = None
        # 가장 최근 embed_texts 호출에서 영구 실패한 텍스트의 원본 인덱스
        # → 호출측(VectorStoreService)이 retry 큐 저장 시 사용
        self._last_failed_indices: list[int] = []
        # 영구 실패 텍스트의 진단 정보 (failed_embeddings JSONL에 저장하기 위함).
        # 인덱스별 dict: {error_type, error_message, text_stats}
        self._last_failed_diagnostics: dict[int, dict[str, object]] = {}
        # 적응형 배치 크기 — 첫 성공한 최대 크기를 기억해 다음 호출에서 재사용
        self._adaptive_batch_size: int | None = None
        # 마지막 임베딩 성공 시각 (epoch 초). 모니터링/stalled 판단용.
        self._last_success_at: float | None = None

    def get_dimension(self) -> int:
        """현재 모델의 벡터 차원 반환

        알려진 모델이면 즉시 반환, 아니면 Ollama에 probe 요청.

        Returns:
            벡터 차원 수

        Raises:
            ConnectionError: Ollama 서버 연결 실패
        """
        if self._cached_dimension is not None:
            return self._cached_dimension

        # 알려진 모델이면 바로 반환
        if self.model in KNOWN_DIMENSIONS:
            self._cached_dimension = KNOWN_DIMENSIONS[self.model]
            return self._cached_dimension

        # 모델명에 알려진 키가 포함되어 있는지 확인 (tag 포함 모델명 대응)
        for known_model, dim in KNOWN_DIMENSIONS.items():
            if known_model in self.model:
                self._cached_dimension = dim
                return self._cached_dimension

        # 알 수 없는 모델 → probe
        logger.info(f"알 수 없는 모델 '{self.model}' — probe 임베딩으로 차원 확인")
        vector = self.embed_text("dimension probe")
        self._cached_dimension = len(vector)
        return self._cached_dimension

    def validate_dimension(self, existing_dim: int) -> None:
        """현재 모델의 벡터 차원이 기존 컬렉션과 호환되는지 검증

        Args:
            existing_dim: 기존 컬렉션의 벡터 차원

        Raises:
            EmbeddingDimensionError: 차원 불일치 시
        """
        current_dim = self.get_dimension()
        if current_dim != existing_dim:
            raise EmbeddingDimensionError(
                f"임베딩 차원 불일치: 현재 모델 '{self.model}'={current_dim}차원, "
                f"기존 컬렉션={existing_dim}차원. "
                f"모델 변경 후에는 기존 케이스를 재인덱싱해야 합니다."
            )

    def embed_text(self, text: str) -> list[float]:
        """단일 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (float 리스트)

        Raises:
            ConnectionError: Ollama 서버 연결 실패
        """
        embeddings = self.get_langchain_embeddings()
        try:
            return embeddings.embed_query(text)
        except Exception as e:
            raise ConnectionError(
                f"Ollama 임베딩 실패 ({self.base_url}, 모델: {self.model}): {e}"
            ) from e

    # 제어 문자 제거용 패턴 (탭 \x09, 줄바꿈 \x0a, 캐리지리턴 \x0d 는 유지)
    _CONTROL_CHAR_RE = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
    )
    # 제로폭 공백, BOM, 기타 보이지 않는 특수 문자
    _INVISIBLE_RE = re.compile(
        r"[\u200b\u200c\u200d\u2060\ufeff\u00ad\u180e"
        r"\u2000-\u200a\u202f\u205f\u3000]"
    )
    # 연속 공백 (줄바꿈 제외)
    _MULTI_SPACE_RE = re.compile(r"[^\S\n]+")

    def _sanitize_text(self, text: str) -> str:
        """임베딩에 안전한 텍스트로 정제

        1. NULL 문자 제거
        2. 제어 문자 제거 (탭/줄바꿈/CR 유지)
        3. 서로게이트 페어 깨진 유니코드 제거
        4. 제로폭 공백, BOM 등 보이지 않는 특수 문자 제거
        5. 연속 공백을 단일 공백으로 치환
        """
        # 서로게이트 깨진 유니코드 제거 (encode → decode with surrogateescape 우회)
        text = text.encode("utf-8", errors="surrogatepass").decode(
            "utf-8", errors="ignore"
        )
        # Cc (제어 문자) 중 유해한 것 제거
        text = self._CONTROL_CHAR_RE.sub("", text)
        # 보이지 않는 유니코드 제거
        text = self._INVISIBLE_RE.sub("", text)
        # 유니코드 카테고리 Cf(포맷 문자) 중 남은 것 제거 (soft hyphen 등)
        text = "".join(
            ch for ch in text
            if unicodedata.category(ch) != "Cf"
        )
        # 연속 공백 → 단일 공백 (줄바꿈 보존)
        text = self._MULTI_SPACE_RE.sub(" ", text)
        return text.strip()

    def embed_texts(
        self,
        texts: list[str],
        cancel_event: threading.Event | None = None,
    ) -> list[list[float]]:
        """배치 텍스트 임베딩 — 동기 래퍼 (내부적으로 async 구현 호출)

        Ollama의 /api/embed 엔드포인트에 여러 배치를 동시 전송 (asyncio.Semaphore로
        embed_max_concurrent 만큼 제한). 호출측 인터페이스는 동기 유지.

        실행 컨텍스트 처리:
        - 일반 동기 컨텍스트: asyncio.run() 직접 호출
        - 이미 이벤트 루프가 도는 환경 (FastAPI 등): ThreadPoolExecutor에서 격리 실행

        Args:
            texts: 임베딩할 텍스트 리스트
            cancel_event: set되면 진행 중인 재시도/분할을 즉시 중단하고 부분 결과 반환

        Returns:
            임베딩 벡터 리스트 (실패한 텍스트는 빈 벡터, 인덱스는 _last_failed_indices)

        Raises:
            ConnectionError: ConnectError가 _CONNECT_MAX_RETRIES회 모두 실패
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.embed_texts_async(texts, cancel_event=cancel_event))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run, self.embed_texts_async(texts, cancel_event=cancel_event)
            )
            return future.result()

    async def embed_texts_async(
        self,
        texts: list[str],
        cancel_event: threading.Event | None = None,
    ) -> list[list[float]]:
        """배치 텍스트 임베딩 (비동기) — 여러 배치를 Ollama에 동시 전송

        httpx.AsyncClient + asyncio.Semaphore(embed_max_concurrent)로 동시성 제한.
        각 배치는 코루틴으로 만들어 asyncio.gather로 병렬 실행.

        시간 제한 (settings.embed_total_timeout_sec, 기본 120s): 호출 시점부터 카운트.
        초과 시 진행 중인 재시도/분할을 즉시 중단하고 성공한 것만 반환 — "어떤 상황에서도
        인덱싱이 N초 이상 같은 배치에 멈춰있으면 안 된다" 원칙.

        적응형 배치 (settings.embed_adaptive_batch=True): 첫 성공한 최대 크기를 기억해
        다음 호출에 재사용 — 256→128→64→32로 분할되면서 안정 크기를 찾으면 유지.

        Args:
            texts: 임베딩할 텍스트 리스트
            cancel_event: set되면 즉시 중단

        Returns:
            임베딩 벡터 리스트 (실패한 텍스트는 빈 벡터, 인덱스는 _last_failed_indices)
        """
        if not texts:
            return []

        # === 1. 텍스트 전처리 (sanitize → 빈 문자열 제거 → 길이/바이트 truncate) ===
        cleaned, skipped, truncated = self._prepare_texts(texts)
        if not cleaned:
            self._last_failed_indices = []
            self._last_failed_diagnostics = {}
            return [[] for _ in texts]

        # === 2. 배치 분할 (적응형 + 길이 편차 관리) ===
        batch_size = self._initial_batch_size()
        max_concurrent = max(1, settings.embed_max_concurrent)
        valid_texts = [t for _, t in cleaned]
        valid_vectors: list[list[float] | None] = [None] * len(valid_texts)
        url = f"{self.base_url}/api/embed"

        batches = self._split_into_batches(valid_texts, batch_size)

        # === 3. 시간 제한 + cancel 관리 ===
        total_start = time.monotonic()
        deadline = total_start + settings.embed_total_timeout_sec
        max_depth = settings.embed_max_subdivide_depth

        # 영구 실패 인덱스 + 진단 정보 (배치 워커들이 동시 추가)
        failed_local_indices: list[int] = []
        failure_diag: dict[int, dict[str, object]] = {}

        semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(
            f"임베딩 동시 전송 시작: {len(batches)}개 배치, 동시성 {max_concurrent} "
            f"(초기 배치 크기 {batch_size}, 시간 제한 {settings.embed_total_timeout_sec}s, "
            f"분할 깊이 {max_depth})"
        )

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            async def run_batch(
                batch_num: int, indices: list[int], batch: list[str]
            ) -> None:
                async with semaphore:
                    if self._should_stop(cancel_event, deadline):
                        self._mark_batch_failed(
                            indices, batch, failed_local_indices, failure_diag,
                            error_type="cancelled" if cancel_event and cancel_event.is_set()
                            else "total_timeout",
                            error_message=(
                                "cancel_event set" if cancel_event and cancel_event.is_set()
                                else f"전체 시간 제한 {settings.embed_total_timeout_sec}s 초과"
                            ),
                        )
                        return
                    batch_start = time.monotonic()
                    await self._embed_with_subdivide_async(
                        client=client,
                        url=url,
                        indices=indices,
                        batch=batch,
                        batch_num=batch_num,
                        results=valid_vectors,
                        failed_local_indices=failed_local_indices,
                        failure_diag=failure_diag,
                        depth=0,
                        deadline=deadline,
                        cancel_event=cancel_event,
                    )
                    elapsed = time.monotonic() - batch_start
                    logger.info(
                        f"배치 {batch_num} 완료: {len(batch)}개 텍스트, {elapsed:.2f}초"
                    )

            await asyncio.gather(
                *(run_batch(bn, idx, b) for bn, idx, b in batches)
            )

        total_elapsed = time.monotonic() - total_start

        # === 4. 결과 재배치 + 통계 + 적응형 배치 갱신 ===
        all_vectors: list[list[float]] = [[] for _ in texts]
        failed_orig_indices: list[int] = []
        failed_orig_diag: dict[int, dict[str, object]] = {}
        for vec_idx, (orig_idx, _) in enumerate(cleaned):
            v = valid_vectors[vec_idx]
            if v is None:
                failed_orig_indices.append(orig_idx)
                if vec_idx in failure_diag:
                    failed_orig_diag[orig_idx] = failure_diag[vec_idx]
                continue
            all_vectors[orig_idx] = v

        if skipped:
            logger.warning(f"빈 텍스트 {skipped}개 스킵 (총 {len(texts)}개 중)")
        if truncated:
            logger.warning(
                f"긴 텍스트 {truncated}개 truncate ({_MAX_CHARS_PER_TEXT}자 또는 "
                f"{settings.embed_max_bytes}B 초과)"
            )

        success_count = len(cleaned) - len(failed_orig_indices)
        if success_count > 0:
            self._last_success_at = time.time()
            # 적응형 배치 — 최소 한 번 성공했으니 현재 batch_size 기억
            if settings.embed_adaptive_batch and self._adaptive_batch_size is None:
                self._adaptive_batch_size = batch_size

        self._last_failed_indices = failed_orig_indices
        self._last_failed_diagnostics = failed_orig_diag
        if failed_orig_indices:
            logger.warning(
                f"임베딩 영구 실패 {len(failed_orig_indices)}개 "
                f"(시간 제한/분할 깊이/cancel 도달)"
            )
        logger.info(
            f"임베딩 완료: {success_count}/{len(cleaned)}개 텍스트 "
            f"(빈 {skipped}개 제외, 실패 {len(failed_orig_indices)}개, "
            f"{len(batches)}개 배치, 총 {total_elapsed:.2f}초)"
        )
        return all_vectors

    # === 내부 헬퍼들 ===

    def _prepare_texts(
        self, texts: list[str]
    ) -> tuple[list[tuple[int, str]], int, int]:
        """sanitize + 빈 문자열 제거 + 문자수/바이트수 truncate

        Returns:
            (cleaned: [(orig_idx, prepared_text)], skipped_empty, truncated_count)
        """
        sanitized_count = 0
        sanitized_texts: list[str] = []
        for t in texts:
            s = self._sanitize_text(t) if t else ""
            if s != (t or ""):
                sanitized_count += 1
            sanitized_texts.append(s)

        if sanitized_count:
            logger.info(f"텍스트 전처리: {sanitized_count}/{len(texts)}개 정제됨")

        max_bytes = settings.embed_max_bytes
        cleaned: list[tuple[int, str]] = []
        truncated = 0
        for idx, t in enumerate(sanitized_texts):
            stripped = t.strip() if t else ""
            if not stripped:
                continue
            was_truncated = False
            # 1차: 문자 수 컷
            if len(stripped) > _MAX_CHARS_PER_TEXT:
                stripped = stripped[:_MAX_CHARS_PER_TEXT]
                was_truncated = True
            # 2차: 바이트 수 컷 (한자/일본어/특수기호로 토큰 폭증 방지). utf-8 기준.
            if max_bytes > 0 and len(stripped.encode("utf-8", errors="ignore")) > max_bytes:
                # 바이트 기준으로 안전하게 자르기 (멀티바이트 경계 깨짐 방지)
                encoded = stripped.encode("utf-8", errors="ignore")[:max_bytes]
                stripped = encoded.decode("utf-8", errors="ignore")
                was_truncated = True
            if was_truncated:
                truncated += 1
            cleaned.append((idx, stripped))

        skipped = len(texts) - len(cleaned)
        return cleaned, skipped, truncated

    def _initial_batch_size(self) -> int:
        """이번 호출에서 사용할 시작 배치 크기 (적응형 갱신 반영)"""
        if settings.embed_adaptive_batch and self._adaptive_batch_size:
            return min(self._adaptive_batch_size, settings.embed_batch_size)
        return settings.embed_batch_size

    def _split_into_batches(
        self, valid_texts: list[str], batch_size: int
    ) -> list[tuple[int, list[int], list[str]]]:
        """배치 분할 — 매우 짧은 텍스트는 별도 그룹, 그룹 내에서 길이 정렬

        attention softmax NaN / padding overhead로 인한 5xx 예방.
        결과 위치 보존을 위해 (batch_num, indices, texts) 튜플로 반환.
        indices[j]는 valid_texts/valid_vectors 내의 원래 위치.

        Returns:
            [(batch_num, original_indices, batch_texts), ...]
        """
        short_threshold = settings.embed_short_text_threshold
        short_items: list[tuple[int, str]] = []  # (orig_idx, text)
        long_items: list[tuple[int, str]] = []
        for i, t in enumerate(valid_texts):
            if len(t) < short_threshold:
                short_items.append((i, t))
            else:
                long_items.append((i, t))

        # 긴 텍스트 그룹 내 편차가 크면 길이순 정렬 → 비슷한 길이끼리 묶이도록
        if long_items and self._has_high_length_variance(long_items):
            long_items.sort(key=lambda x: len(x[1]))
            logger.debug(
                f"배치 길이 편차 큼 → 길이 정렬 적용 ({len(long_items)}개 텍스트)"
            )

        result: list[tuple[int, list[int], list[str]]] = []
        batch_num = 0
        for group in (short_items, long_items):
            for i in range(0, len(group), batch_size):
                slice_ = group[i : i + batch_size]
                if not slice_:
                    continue
                batch_num += 1
                result.append((
                    batch_num,
                    [idx for idx, _ in slice_],
                    [t for _, t in slice_],
                ))
        return result

    @staticmethod
    def _has_high_length_variance(items: list[tuple[int, str]]) -> bool:
        """배치 내 최장/최단 텍스트 길이 비가 10배 이상인지"""
        if len(items) < 2:
            return False
        lengths = [len(t) for _, t in items]
        return max(lengths) >= min(lengths) * 10

    def _should_stop(
        self,
        cancel_event: threading.Event | None,
        deadline: float,
    ) -> bool:
        """cancel 또는 시간 제한 도달 여부"""
        if cancel_event and cancel_event.is_set():
            return True
        return time.monotonic() >= deadline

    def _mark_batch_failed(
        self,
        indices: list[int],
        batch: list[str],
        failed_local_indices: list[int],
        failure_diag: dict[int, dict[str, object]],
        error_type: str,
        error_message: str,
    ) -> None:
        """배치 통째로 영구 실패 처리 + 진단 정보 기록"""
        for j, text in enumerate(batch):
            idx = indices[j]
            failed_local_indices.append(idx)
            failure_diag[idx] = {
                "error_type": error_type,
                "error_message": error_message,
                "text_stats": text_stats(text),
            }

    async def _embed_with_subdivide_async(
        self,
        client: httpx.AsyncClient,
        url: str,
        indices: list[int],
        batch: list[str],
        batch_num: int,
        results: list[list[float] | None],
        failed_local_indices: list[int],
        failure_diag: dict[int, dict[str, object]],
        depth: int,
        deadline: float,
        cancel_event: threading.Event | None,
    ) -> None:
        """배치 임베딩 + 실패 시 절반으로 쪼개 재시도 (async)

        깊이 제한 (settings.embed_max_subdivide_depth, 기본 3): 도달 시 영구 실패.
        모든 진입점에서 cancel/시간 제한 체크.

        실패 사유가 거대 텍스트 1개 때문이면 분할로 고립시킬 수 있고, 나머지는
        정상 임베딩됨. 그러나 무한 분할은 5일 멈춤의 원인이었으므로 깊이 제한 필수.
        """
        # 진입 시 cancel/시간 제한 체크
        if self._should_stop(cancel_event, deadline):
            self._mark_batch_failed(
                indices, batch, failed_local_indices, failure_diag,
                error_type="cancelled" if cancel_event and cancel_event.is_set()
                else "total_timeout",
                error_message=(
                    "cancel_event set during subdivide"
                    if cancel_event and cancel_event.is_set()
                    else f"전체 시간 제한 {settings.embed_total_timeout_sec}s 초과"
                ),
            )
            return

        # 분할 깊이 초과 → 영구 실패 (각 청크 진단 정보와 함께)
        max_depth = settings.embed_max_subdivide_depth
        if depth >= max_depth:
            for j, text in enumerate(batch):
                idx = indices[j]
                failed_local_indices.append(idx)
                stats = text_stats(text)
                failure_diag[idx] = {
                    "error_type": "subdivide_exhausted",
                    "error_message": f"분할 깊이 {max_depth} 도달, 배치 크기 {len(batch)}",
                    "text_stats": stats,
                }
                logger.error(
                    f"독성 청크: {_format_text_stats(stats)}, "
                    f"batch={batch_num}#{j}, depth={depth}, "
                    f"앞부분={stats['preview']!r}"
                )
            return

        try:
            embeddings, error_type = await self._embed_batch_with_retry_async(
                client, url, batch, batch_num, deadline, cancel_event
            )
            for j, vec in enumerate(embeddings):
                results[indices[j]] = vec
            return
        except ConnectionError as e:
            # 분할 종착 — 1개 텍스트도 실패 → 영구 실패로 기록
            if len(batch) <= _MIN_SUBDIVIDE_SIZE:
                idx = indices[0]
                stats = text_stats(batch[0]) if batch else {}
                failed_local_indices.append(idx)
                failure_diag[idx] = {
                    "error_type": getattr(e, "_embed_error_type", "unknown"),
                    "error_message": str(e),
                    "text_stats": stats,
                }
                logger.error(
                    f"단일 텍스트 임베딩 실패 (batch {batch_num}, "
                    f"{_format_text_stats(stats) if stats else 'empty'}): {e}"
                )
                logger.error(
                    f"독성 청크 앞부분: {stats.get('preview', '')!r}"
                )
                return

            # 분할 후 재귀
            mid = len(batch) // 2
            logger.warning(
                f"배치 {batch_num} 실패 (depth={depth}, type={getattr(e, '_embed_error_type', '?')}) "
                f"→ {len(batch)}개를 {mid}+{len(batch) - mid}개로 분할 재시도"
            )
            await self._embed_with_subdivide_async(
                client, url, indices[:mid], batch[:mid], batch_num * 10 + 1,
                results, failed_local_indices, failure_diag,
                depth + 1, deadline, cancel_event,
            )
            await self._embed_with_subdivide_async(
                client, url, indices[mid:], batch[mid:], batch_num * 10 + 2,
                results, failed_local_indices, failure_diag,
                depth + 1, deadline, cancel_event,
            )

    async def _embed_batch_with_retry_async(
        self,
        client: httpx.AsyncClient,
        url: str,
        batch: list[str],
        batch_num: int,
        deadline: float,
        cancel_event: threading.Event | None,
    ) -> tuple[list[list[float]], str]:
        """단일 배치 임베딩 (async) — 에러 유형별로 다른 재시도 정책

        - timeout (httpx.TimeoutException/ReadTimeout): 재시도 0회 즉시 raise
          → subdivide로 빠르게 넘어감. 같은 크기로 재시도해봐야 또 timeout.
        - HTTP 5xx: 1회만 짧게 재시도 (2초). bge-m3 5xx는 보통 토큰 폭증 / NaN /
          OOM이라 같은 입력 재시도는 의미 없음, subdivide가 진짜 해결책.
        - HTTP 4xx: 즉시 raise (입력 문제는 재시도해도 동일).
        - ConnectError/NetworkError: 3회 exponential backoff (2/4/8s).

        Returns:
            (embeddings, success_marker) — 성공한 경우만.

        Raises:
            ConnectionError with attribute _embed_error_type:
                "timeout" | "http_500" | "http_4xx" | "connection"
        """
        payload = {
            "model": self.model,
            "input": batch,
            "keep_alive": -1,
        }

        # === HTTP 호출을 한 번 시도하고 결과/에러 분류 ===
        async def attempt_post() -> tuple[list[list[float]], None] | tuple[None, tuple[str, Exception]]:
            try:
                resp = await client.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.json()["embeddings"], None
            except httpx.HTTPStatusError as he:
                code = he.response.status_code
                if 400 <= code < 500:
                    return None, ("http_4xx", he)
                return None, ("http_500", he)
            except (httpx.TimeoutException, httpx.ReadTimeout) as te:
                return None, ("timeout", te)
            except (
                httpx.ConnectError, httpx.ReadError,
                httpx.RemoteProtocolError, httpx.NetworkError,
            ) as ne:
                return None, ("connection", ne)

        # === ConnectError 전용: 3회 exponential backoff ===
        connect_attempts = 0
        server_error_attempts = 0
        last_err: Exception | None = None
        last_type: str = "unknown"

        while True:
            # 매 반복 시작 시 cancel/deadline 체크
            if self._should_stop(cancel_event, deadline):
                err = ConnectionError(
                    f"임베딩 중단 (batch {batch_num}, "
                    f"{'cancel' if cancel_event and cancel_event.is_set() else 'timeout'})"
                )
                err._embed_error_type = (
                    "cancelled" if cancel_event and cancel_event.is_set() else "total_timeout"
                )
                raise err

            embeddings, err_info = await attempt_post()
            if embeddings is not None:
                return embeddings, "ok"

            err_type, err_obj = err_info
            last_err = err_obj
            last_type = err_type

            if err_type == "http_4xx":
                # 4xx는 즉시 실패 (재시도해도 같은 결과)
                err = ConnectionError(
                    f"Ollama 배치 임베딩 실패 (batch {batch_num}, HTTP "
                    f"{err_obj.response.status_code if hasattr(err_obj, 'response') else '?'}, "
                    f"{self.base_url}, 모델: {self.model}): {err_obj}"
                )
                err._embed_error_type = "http_4xx"
                raise err from err_obj

            if err_type == "timeout":
                # timeout은 즉시 실패 → subdivide가 처리
                err = ConnectionError(
                    f"Ollama 배치 임베딩 timeout (batch {batch_num}, 크기 {len(batch)}, "
                    f"{_REQUEST_TIMEOUT}s 초과): {err_obj}"
                )
                err._embed_error_type = "timeout"
                raise err from err_obj

            if err_type == "http_500":
                if server_error_attempts == 0:
                    server_error_attempts += 1
                    logger.warning(
                        f"임베딩 5xx 1회 재시도 (batch {batch_num}, "
                        f"{_SERVER_ERROR_RETRY_WAIT:.0f}s 후): {err_obj}"
                    )
                    await asyncio.sleep(_SERVER_ERROR_RETRY_WAIT)
                    continue
                err = ConnectionError(
                    f"Ollama 배치 임베딩 5xx 2회 실패 (batch {batch_num}, "
                    f"크기 {len(batch)}): {err_obj}"
                )
                err._embed_error_type = "http_500"
                raise err from err_obj

            # err_type == "connection"
            if connect_attempts < _CONNECT_MAX_RETRIES - 1:
                connect_attempts += 1
                wait = _CONNECT_RETRY_BACKOFF_BASE ** connect_attempts
                logger.warning(
                    f"임베딩 connection 재시도 {connect_attempts}/{_CONNECT_MAX_RETRIES - 1} "
                    f"(batch {batch_num}, {wait:.0f}s 후): {err_obj}"
                )
                # backoff 동안에도 cancel/deadline 체크 가능하도록 짧게 sleep + 체크 반복
                slept = 0.0
                while slept < wait:
                    if self._should_stop(cancel_event, deadline):
                        err = ConnectionError(
                            f"임베딩 중단 (batch {batch_num}, backoff 중)"
                        )
                        err._embed_error_type = (
                            "cancelled" if cancel_event and cancel_event.is_set()
                            else "total_timeout"
                        )
                        raise err
                    step = min(0.5, wait - slept)
                    await asyncio.sleep(step)
                    slept += step
                continue

            err = ConnectionError(
                f"Ollama connection {_CONNECT_MAX_RETRIES}회 실패 "
                f"(batch {batch_num}, {self.base_url}, 모델: {self.model}): {err_obj}"
            )
            err._embed_error_type = "connection"
            raise err from err_obj

    def get_langchain_embeddings(self):
        """LangChain 호환 임베딩 객체 반환

        Returns:
            OllamaEmbeddings 인스턴스 (LangChain 호환)
        """
        if self._langchain_embeddings is None:
            from langchain_ollama import OllamaEmbeddings

            self._langchain_embeddings = OllamaEmbeddings(
                model=self.model,
                base_url=self.base_url,
            )

        return self._langchain_embeddings
