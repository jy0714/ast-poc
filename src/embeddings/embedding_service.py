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
import time
import unicodedata

import httpx

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 재시도 설정 (일시적 timeout / 네트워크 hiccup 대응)
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2.0  # 초 (2 → 4 → 8)
_REQUEST_TIMEOUT = 60.0  # 1분 — 빨리 실패하고 작은 배치로 재시도하는 편이 전체 처리량에 유리

# bge-m3 컨텍스트 한도(`/api/ps` 응답상 4096 토큰)에 맞춘 보수적 절단.
# 한글 평균 1자 ≈ 1토큰, 영문 1자 ≈ 0.3토큰. 헤더/마진 고려 2500자에서 절단.
_MAX_CHARS_PER_TEXT = 2500

# 배치 실패 시 binary subdivide 최소 단위 — 이 이하로는 더 못 쪼갬
_MIN_SUBDIVIDE_SIZE = 1

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

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """배치 텍스트 임베딩 — 동기 래퍼 (내부적으로 async 구현 호출)

        Ollama의 /api/embed 엔드포인트에 여러 배치를 동시 전송 (asyncio.Semaphore로
        embed_max_concurrent 만큼 제한). 호출측 인터페이스는 동기 유지.

        실행 컨텍스트 처리:
        - 일반 동기 컨텍스트: asyncio.run() 직접 호출
        - 이미 이벤트 루프가 도는 환경 (FastAPI 등): ThreadPoolExecutor에서 격리 실행

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트

        Raises:
            ConnectionError: Ollama 서버 연결 실패
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 실행 중인 루프 없음 → asyncio.run() 으로 직접 실행
            return asyncio.run(self.embed_texts_async(texts))

        # 이미 루프 위에서 호출됨 → 별도 스레드에서 새 루프로 실행
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self.embed_texts_async(texts))
            return future.result()

    async def embed_texts_async(self, texts: list[str]) -> list[list[float]]:
        """배치 텍스트 임베딩 (비동기) — 여러 배치를 Ollama에 동시 전송

        httpx.AsyncClient + asyncio.Semaphore(embed_max_concurrent)로 동시성 제한.
        각 배치는 코루틴으로 만들어 asyncio.gather로 병렬 실행.
        실패 시 binary subdivide 재시도 로직은 동기 버전과 동일.

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트

        Raises:
            ConnectionError: Ollama 서버 연결 실패
        """
        if not texts:
            return []

        # 텍스트 전처리 (sanitize) → Ollama에 안전한 입력으로 정제
        sanitized_count = 0
        sanitized_texts: list[str] = []
        for t in texts:
            s = self._sanitize_text(t) if t else ""
            if s != (t or ""):
                sanitized_count += 1
            sanitized_texts.append(s)

        if sanitized_count:
            logger.info(f"텍스트 전처리: {sanitized_count}/{len(texts)}개 정제됨")

        # 빈 문자열/공백만 있는 텍스트를 필터링 (Ollama 400 에러 방지)
        # + 모델 토큰 한도를 넘는 텍스트는 truncate (bge-m3 4096 토큰 한도)
        cleaned: list[tuple[int, str]] = []
        truncated = 0
        for idx, t in enumerate(sanitized_texts):
            stripped = t.strip() if t else ""
            if not stripped:
                continue
            if len(stripped) > _MAX_CHARS_PER_TEXT:
                stripped = stripped[:_MAX_CHARS_PER_TEXT]
                truncated += 1
            cleaned.append((idx, stripped))

        if not cleaned:
            return [[] for _ in texts]

        batch_size = settings.embed_batch_size
        max_concurrent = max(1, settings.embed_max_concurrent)
        # 유효 텍스트만 임베딩
        valid_texts = [t for _, t in cleaned]
        valid_vectors: list[list[float] | None] = [None] * len(valid_texts)
        url = f"{self.base_url}/api/embed"

        # 배치 단위로 분할
        batches: list[tuple[int, int, list[str]]] = []  # (batch_num, local_offset, batch)
        for i in range(0, len(valid_texts), batch_size):
            batches.append((i // batch_size + 1, i, valid_texts[i : i + batch_size]))

        failed_local_indices: list[int] = []
        semaphore = asyncio.Semaphore(max_concurrent)
        total_start = time.monotonic()
        logger.info(
            f"임베딩 동시 전송 시작: {len(batches)}개 배치, "
            f"동시성 {max_concurrent} (배치 크기 {batch_size})"
        )

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            async def run_batch(batch_num: int, local_offset: int, batch: list[str]) -> None:
                async with semaphore:
                    batch_start = time.monotonic()
                    await self._embed_with_subdivide_async(
                        client=client,
                        url=url,
                        batch=batch,
                        batch_num=batch_num,
                        local_offset=local_offset,
                        results=valid_vectors,
                        failed_local_indices=failed_local_indices,
                    )
                    elapsed = time.monotonic() - batch_start
                    logger.info(
                        f"배치 {batch_num} 완료: {len(batch)}개 텍스트, {elapsed:.2f}초"
                    )

            await asyncio.gather(
                *(run_batch(bn, lo, b) for bn, lo, b in batches)
            )

        total_elapsed = time.monotonic() - total_start

        # 원래 인덱스에 맞게 결과 재배치 (빈 텍스트 → 빈 벡터, 실패 텍스트 → 빈 벡터)
        all_vectors: list[list[float]] = [[] for _ in texts]
        failed_orig_indices: list[int] = []
        for vec_idx, (orig_idx, _) in enumerate(cleaned):
            v = valid_vectors[vec_idx]
            if v is None:
                failed_orig_indices.append(orig_idx)
                continue
            all_vectors[orig_idx] = v

        skipped = len(texts) - len(cleaned)
        if skipped:
            logger.warning(f"빈 텍스트 {skipped}개 스킵 (총 {len(texts)}개 중)")
        if truncated:
            logger.warning(
                f"긴 텍스트 {truncated}개 truncate ({_MAX_CHARS_PER_TEXT}자 초과)"
            )
        if failed_orig_indices:
            # 실패한 텍스트 인덱스를 인스턴스 속성으로 노출 → 호출측이 retry 큐에 저장 가능
            self._last_failed_indices = failed_orig_indices
            logger.warning(
                f"임베딩 영구 실패 {len(failed_orig_indices)}개 (binary subdivide 후에도 실패)"
            )
        else:
            self._last_failed_indices = []
        logger.info(
            f"임베딩 완료: {len(cleaned) - len(failed_orig_indices)}/{len(cleaned)}개 텍스트 "
            f"(빈 {skipped}개 제외, 실패 {len(failed_orig_indices)}개, "
            f"{len(batches)}개 배치, 총 {total_elapsed:.2f}초)"
        )
        return all_vectors

    async def _embed_with_subdivide_async(
        self,
        client: httpx.AsyncClient,
        url: str,
        batch: list[str],
        batch_num: int,
        local_offset: int,
        results: list[list[float] | None],
        failed_local_indices: list[int],
    ) -> None:
        """배치 임베딩 + 실패 시 절반으로 쪼개 재시도 (async)

        실패 사유가 거대 텍스트 1개 때문이면 binary subdivide로 그 텍스트만
        고립시킬 수 있고, 나머지 텍스트는 정상 임베딩됨.
        """
        try:
            embeddings = await self._embed_batch_with_retry_async(
                client, url, batch, batch_num
            )
            for j, vec in enumerate(embeddings):
                results[local_offset + j] = vec
            return
        except ConnectionError as e:
            if len(batch) <= _MIN_SUBDIVIDE_SIZE:
                # 1개 텍스트도 임베딩 실패 → 영구 실패로 기록
                logger.error(
                    f"단일 텍스트 임베딩 실패 (batch {batch_num}, "
                    f"길이={len(batch[0]) if batch else 0}자): {e}"
                )
                for j in range(len(batch)):
                    failed_local_indices.append(local_offset + j)
                return

            mid = len(batch) // 2
            logger.warning(
                f"배치 {batch_num} 실패 → {len(batch)}개를 {mid}+{len(batch) - mid}개로 분할 재시도"
            )
            await self._embed_with_subdivide_async(
                client, url, batch[:mid], batch_num * 10 + 1,
                local_offset, results, failed_local_indices,
            )
            await self._embed_with_subdivide_async(
                client, url, batch[mid:], batch_num * 10 + 2,
                local_offset + mid, results, failed_local_indices,
            )

    async def _embed_batch_with_retry_async(
        self,
        client: httpx.AsyncClient,
        url: str,
        batch: list[str],
        batch_num: int,
    ) -> list[list[float]]:
        """단일 배치 임베딩 (async) — 일시적 timeout/네트워크 에러 시 재시도

        keep_alive=-1로 모델을 VRAM에 상주시켜 배치 간 재로드 비용을 제거.
        timeout/connect 에러는 _MAX_RETRIES회까지 지수 백오프로 재시도.
        HTTP 4xx (입력 문제)는 즉시 실패.
        """
        payload = {
            "model": self.model,
            "input": batch,
            "keep_alive": -1,  # 모델을 VRAM에 무기한 유지
        }

        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await client.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.json()["embeddings"]

            except httpx.HTTPStatusError as e:
                # 4xx는 재시도해도 같은 결과 → 즉시 실패
                if 400 <= e.response.status_code < 500:
                    raise ConnectionError(
                        f"Ollama 배치 임베딩 실패 (batch {batch_num}, "
                        f"HTTP {e.response.status_code}, {self.base_url}, "
                        f"모델: {self.model}): {e}"
                    ) from e
                last_err = e

            except (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.NetworkError,
            ) as e:
                last_err = e

            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    f"임베딩 재시도 {attempt + 1}/{_MAX_RETRIES - 1} "
                    f"(batch {batch_num}, {wait:.0f}초 후): {last_err}"
                )
                await asyncio.sleep(wait)

        raise ConnectionError(
            f"Ollama 배치 임베딩 {_MAX_RETRIES}회 재시도 모두 실패 "
            f"(batch {batch_num}, {self.base_url}, 모델: {self.model}): {last_err}"
        ) from last_err

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
