"""Reranker — 검색 후보를 질의와의 관련성으로 재정렬 (v4, 지시서 §5.4)

백엔드 (`RERANK_BACKEND`):
- qwen3 (서비스 기본): transformers `Qwen3-Reranker`. system/user/assistant 템플릿에
  <Instruct>/<Query>/<Document>를 넣고 마지막 위치의 yes/no 로짓으로 P(yes)를 점수로 쓴다.
  `logits_to_keep=1`로 마지막 위치 로짓만 계산한다(어휘 15만 × 길이 로짓을 만들지 않게).
  개발 PC(Turing, BF16 미지원)는 `Qwen/Qwen3-Reranker-0.6B` + `RERANK_DTYPE=fp16`.
- bge (R9 비교 실험 전용): FlagEmbedding `bge-reranker-v2-m3`.
- AST_FAKE_MODELS=1이면 백엔드와 관계없이 토큰 겹침 기반 가짜 재순위기를 쓴다.

두 실제 백엔드는 같은 `RERANK_MAX_LENGTH`(4,096)를 쓰고(R9 공정성), 문서가 길어 잘린
건수를 `n_truncated`에 누적한다(논문 4.6절 절단 비율 근거).

torch·transformers·FlagEmbedding은 첫 점수 계산 때 지연 로드한다. 프로세스당 1회만
적재한다(get_reranker 싱글턴).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any, Protocol

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 모듈 레벨 캐시 — (백엔드, 모델)마다 1개. R8(qwen3)·R9(bge)를 번갈아 돌려도 다시 적재하지 않는다.
_instances: dict[tuple[str, str], Reranker] = {}
_instance_lock = threading.Lock()

# Qwen3-Reranker 템플릿 (HF 모델 카드 형식)
RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


class RerankerLoadError(Exception):
    """Reranker 모델 로드 실패"""


def format_rerank_pair(query: str, doc: str, instruction: str | None = None) -> str:
    """Qwen3-Reranker 사용자 메시지 본문 — `<Instruct>: … <Query>: … <Document>: …`"""
    inst = instruction if instruction is not None else settings.rerank_instruction
    return f"<Instruct>: {inst}\n<Query>: {query}\n<Document>: {doc}"


class RerankBackend(Protocol):
    """재순위 백엔드 공통 인터페이스"""

    name: str
    model_name: str
    n_scored: int
    n_truncated: int

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        """(질의, 문서) 쌍마다 0~1 관련도 점수"""
        ...


def _resolve_device() -> str:
    """CUDA가 있으면 cuda, 없으면 cpu"""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Qwen3RerankBackend:
    """transformers 기반 Qwen3-Reranker (ref/qwen3_formats.Qwen3Reranker 이식)"""

    name = "qwen3"

    def __init__(
        self,
        model_name: str | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
        dtype: str | None = None,
        instruction: str | None = None,
    ) -> None:
        """설정만 저장한다. 모델은 첫 score() 호출 때 적재한다.

        Args:
            model_name: HF 모델 이름 (기본 settings.rerank_model)
            max_length: 템플릿·질의·문서 합계 토큰 상한 (기본 settings.rerank_max_length)
            batch_size: 배치 크기 (기본 settings.rerank_batch_size)
            dtype: bf16 | fp16 | fp32 (기본 settings.rerank_dtype)
            instruction: 재순위 지시문 (기본 settings.rerank_instruction)
        """
        self.model_name = model_name or settings.rerank_model
        self.max_length = max_length or settings.rerank_max_length
        self.batch_size = batch_size or settings.rerank_batch_size
        self.dtype = (dtype or settings.rerank_dtype).lower()
        self.instruction = instruction if instruction is not None else settings.rerank_instruction
        self.n_scored = 0
        self.n_truncated = 0
        self.tok: Any = None
        self.model: Any = None
        self.device = "cpu"
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        """토크나이저·모델 지연 적재"""
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            if "qwen3-reranker" not in self.model_name.lower():
                raise RerankerLoadError(
                    f"RERANK_BACKEND=qwen3인데 RERANK_MODEL이 Qwen3-Reranker가 아닙니다: "
                    f"{self.model_name}. .env의 RERANK_MODEL을 확인해 주세요 "
                    f"(bge 모델은 RERANK_BACKEND=bge + RERANK_BGE_MODEL)."
                )
            if self.dtype not in _DTYPES:
                raise RerankerLoadError(f"RERANK_DTYPE은 bf16|fp16|fp32 중 하나: {self.dtype}")
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as e:
                raise RerankerLoadError(
                    "torch·transformers가 설치되지 않았습니다 (Python 3.12 venv 확인)."
                ) from e

            logger.info(f"Qwen3 Reranker 로딩: {self.model_name} ({self.dtype})")
            t0 = time.time()
            try:
                self.device = _resolve_device()
                self.tok = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=getattr(torch, _DTYPES[self.dtype]),
                    attn_implementation="sdpa",
                )
                self.model = model.to(self.device).eval()
            except Exception as e:
                raise RerankerLoadError(f"Qwen3 Reranker 로드 실패 ({self.model_name}): {e}") from e
            self.yes_id = self.tok.convert_tokens_to_ids("yes")
            self.no_id = self.tok.convert_tokens_to_ids("no")
            self.prefix_ids = self.tok.encode(RERANK_PREFIX, add_special_tokens=False)
            self.suffix_ids = self.tok.encode(RERANK_SUFFIX, add_special_tokens=False)
            logger.info(f"Qwen3 Reranker 로딩 완료: {time.time() - t0:.1f}s (device={self.device})")

    def encode(self, query: str, doc: str) -> list[int]:
        """한 쌍의 입력 토큰 ID. 상한을 넘으면 문서 끝을 자르고 n_truncated를 올린다."""
        budget = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        head = self.tok.encode(
            format_rerank_pair(query, "", self.instruction), add_special_tokens=False
        )
        doc_ids = self.tok.encode(doc, add_special_tokens=False)
        room = max(0, budget - len(head))
        if len(doc_ids) > room:
            doc_ids = doc_ids[:room]
            self.n_truncated += 1
        self.n_scored += 1
        return self.prefix_ids + head + doc_ids + self.suffix_ids

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        """P(yes) 점수 (0~1). 입력 순서를 유지한다."""
        if not docs:
            return []
        self._load()
        torch = __import__("torch")
        scores: list[float] = []
        for i in range(0, len(docs), self.batch_size):
            ids = [self.encode(query, d) for d in docs[i : i + self.batch_size]]
            batch = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
            pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1).float()
            scores.extend(torch.softmax(pair, dim=1)[:, 1].tolist())
        return scores


class BgeRerankBackend:
    """FlagEmbedding bge-reranker-v2-m3 (R9 비교 실험 전용)"""

    name = "bge"

    def __init__(self, model_name: str | None = None, max_length: int | None = None) -> None:
        """설정만 저장한다. 모델은 첫 score() 호출 때 적재한다.

        Args:
            model_name: HF 모델 이름 (기본 settings.rerank_bge_model)
            max_length: 쌍 토큰 상한 (기본 settings.rerank_max_length — FlagEmbedding 기본 512를 쓰지 않음)
        """
        self.model_name = model_name or settings.rerank_bge_model
        self.max_length = max_length or settings.rerank_max_length
        self.n_scored = 0
        self.n_truncated = 0
        self.model: Any = None
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        """FlagReranker 지연 적재"""
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as e:
                raise RerankerLoadError(
                    "FlagEmbedding 패키지가 설치되지 않았습니다. "
                    "pip install FlagEmbedding 으로 설치해 주세요."
                ) from e
            logger.info(f"bge Reranker 로딩: {self.model_name}")
            t0 = time.time()
            try:
                self.model = FlagReranker(
                    self.model_name, use_fp16=True, max_length=self.max_length
                )
            except Exception as e:
                raise RerankerLoadError(f"Reranker 모델 로드 실패 ({self.model_name}): {e}") from e
            logger.info(f"bge Reranker 로딩 완료: {time.time() - t0:.1f}s")

    def _count_truncated(self, query: str, docs: Sequence[str]) -> int:
        """max_length를 넘는 쌍의 수 (토크나이저를 못 쓰면 0)"""
        tok = getattr(self.model, "tokenizer", None)
        if tok is None:
            return 0
        try:
            enc = tok([query] * len(docs), list(docs), add_special_tokens=True)["input_ids"]
        except Exception:
            return 0
        return sum(1 for ids in enc if len(ids) > self.max_length)

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        """시그모이드 정규화 점수 (0~1). 입력 순서를 유지한다."""
        if not docs:
            return []
        self._load()
        self.n_truncated += self._count_truncated(query, docs)
        self.n_scored += len(docs)
        scores = self.model.compute_score(
            [[query, d] for d in docs], normalize=True, max_length=self.max_length
        )
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [float(s) for s in scores]


class FakeRerankBackend:
    """토큰 겹침 기반 가짜 재순위기 (AST_FAKE_MODELS=1)

    점수 = |질의 토큰 ∩ 문서 토큰| / |질의 토큰| (BM25 토크나이저 기준). 결정적이다.
    """

    name = "fake"
    model_name = "fake-overlap"

    def __init__(self) -> None:
        """카운터 초기화"""
        self.n_scored = 0
        self.n_truncated = 0

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        """질의 토큰이 문서에 들어 있는 비율"""
        from src.text.bm25_tokenizer import tokenize

        q = set(tokenize(query))
        self.n_scored += len(docs)
        if not q:
            return [0.0 for _ in docs]
        return [len(q & set(tokenize(d))) / len(q) for d in docs]


def make_backend(backend: str | None = None, model_name: str | None = None) -> RerankBackend:
    """설정(또는 인자)에 맞는 백엔드 생성

    Args:
        backend: "qwen3" | "bge" (기본 settings.rerank_backend)
        model_name: 모델 이름 덮어쓰기

    Returns:
        재순위 백엔드 (모델 적재 전)
    """
    if settings.ast_fake_models:
        return FakeRerankBackend()
    name = (backend or settings.rerank_backend).lower()
    if name == "qwen3":
        return Qwen3RerankBackend(model_name=model_name)
    if name == "bge":
        return BgeRerankBackend(model_name=model_name)
    raise ValueError(f"RERANK_BACKEND는 qwen3|bge 중 하나: {name}")


class Reranker:
    """재순위기 — 백엔드를 감싸 점수 계산과 (서비스용) 정렬을 제공

    사용법:
        reranker = get_reranker()
        scores = reranker.score("질의", ["문서1", "문서2"])     # retrieve()가 쓰는 경로
        reranked = reranker.rerank(query="질의", documents=[...], top_n=5)  # 기존 엔진 호환
    """

    def __init__(self, backend: str | None = None, model_name: str | None = None) -> None:
        """Reranker 초기화 (모델은 첫 호출 시 지연 로딩)

        Args:
            backend: "qwen3" | "bge" (기본 settings.rerank_backend)
            model_name: 모델 이름 덮어쓰기 (기본: 백엔드별 설정값)
        """
        self.backend: RerankBackend = make_backend(backend, model_name)
        self.backend_name = self.backend.name
        self.model_name = self.backend.model_name

    @property
    def n_truncated(self) -> int:
        """지금까지 max_length를 넘어 잘린 쌍의 수"""
        return self.backend.n_truncated

    @property
    def n_scored(self) -> int:
        """지금까지 점수를 낸 쌍의 수"""
        return self.backend.n_scored

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        """(질의, 문서) 쌍마다 0~1 점수. 입력 순서를 유지한다.

        Args:
            query: 질의 원문 (지시문은 백엔드가 붙인다)
            docs: 재순위 입력 텍스트 (헤더 on 프로필이면 header + "\\n\\n" + text)

        Returns:
            점수 리스트 (docs와 같은 길이)
        """
        if not docs:
            return []
        t0 = time.time()
        scores = self.backend.score(query, docs)
        logger.info(f"Reranker({self.backend_name}) {len(docs)}건: {time.time() - t0:.2f}s")
        return scores

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """검색 결과를 query와의 관련성으로 재정렬 (기존 엔진 경로 호환)

        Args:
            query: 사용자 질의
            documents: 각 dict에 "content" 키가 있는 검색 결과
            top_n: 반환할 최대 결과 수 (기본: settings.rerank_top_n)
            min_score: 이 점수 미만 결과 제거 (기본: settings.rerank_min_score).
                       top_n 선별 후 적용.

        Returns:
            rerank_score가 추가된 dict 리스트 (점수 내림차순, 동점은 입력 순서).
            필터 후 0건이면 빈 리스트.
        """
        if not documents:
            return []

        top_n = top_n or settings.rerank_top_n
        threshold = min_score if min_score is not None else settings.rerank_min_score

        scores = self.score(query, [doc.get("content", "") for doc in documents])

        scored_docs = []
        for doc, score in zip(documents, scores, strict=True):
            enriched = dict(doc)
            enriched["rerank_score"] = float(score)
            scored_docs.append(enriched)
        # sort는 안정 정렬이라 동점이면 입력(RRF) 순서를 유지한다
        scored_docs.sort(key=lambda d: d["rerank_score"], reverse=True)

        top_docs = scored_docs[:top_n]
        filtered = [d for d in top_docs if d["rerank_score"] >= threshold]
        if len(filtered) < len(top_docs):
            logger.info(
                f"Reranker 필터링: {len(top_docs)}건 중 {len(filtered)}건 통과 "
                f"(min_score={threshold})"
            )
        return filtered


def get_reranker(backend: str | None = None, model_name: str | None = None) -> Reranker:
    """(백엔드, 모델)별 싱글톤 Reranker 반환

    Args:
        backend: "qwen3" | "bge" (None이면 settings 기본값)
        model_name: 모델명 (None이면 백엔드별 settings 기본값)

    Returns:
        Reranker 인스턴스 (캐싱됨, 모델은 첫 score 때 적재)
    """
    with _instance_lock:
        # 백엔드 생성은 설정 저장뿐이라 비용이 없다
        candidate = Reranker(backend=backend, model_name=model_name)
        key = (candidate.backend_name, candidate.model_name)
        inst = _instances.get(key)
        if inst is None:
            _instances[key] = inst = candidate
        return inst


def release_rerankers() -> None:
    """캐시된 재순위기를 모두 버린다 (VRAM 반환은 GC·torch에 맡김)"""
    with _instance_lock:
        _instances.clear()
