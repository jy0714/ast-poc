"""검색 함수 retrieve() — 서비스와 실험이 공용으로 쓴다 (지시서 §5.6)

알고리즘
1. 필터: filters가 있으면 파트 마스크 (실험에서는 항상 None)
2. BM25: 질의를 bm25_tokenizer로 처리해 파트 점수(> 0만) → 단위별 첫 등장 파트를 대표로 단위 순위
3. Dense: 질의 임베딩(Qwen3면 지시문) → 파트 코사인 → 같은 방식으로 단위 순위
4. 단일 검색기: 그 목록을 depth까지 반환
5. RRF: 검색기별 상위 k 단위에 score(u) = Σ 1/(rrf_k + rank). 동점이면 두 검색기 중 더 좋은
   순위, 그다음 unit_id. 대표 파트는 RRF 기여가 더 큰 검색기의 대표 파트
6. 재순위: 상위 rerank_candidates 단위를 재순위 점수 내림차순(동점은 RRF 순서), 그 뒤에
   나머지를 원래 순서로 붙여 depth까지
7. 점수 하한·가중치·상대 필터는 쓰지 않는다

UnitHit.score는 그 순위를 정한 단계의 점수다(재순위 구간은 P(yes), 꼬리는 RRF 점수,
단일 검색기는 BM25/코사인). 단계마다 척도가 다르므로 TREC 런 파일에는 순위 기반 점수를 쓴다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.embeddings.embedding_service import EmbeddingService
from src.index import case_index
from src.index.case_index import CaseIndex, embed_input, profile_spec
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievalConfig:
    """검색 설정 — 기본값은 지시서 §5.6 고정값 (평가 결과를 보고 바꾸지 않는다)"""

    profile: str = "hdr-qwen3"
    use_bm25: bool = True
    use_dense: bool = True
    rerank: bool = True
    reranker: str = "qwen3"  # "qwen3" | "bge"
    k_bm25: int = 50
    k_dense: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 30
    depth: int = 100


@dataclass
class UnitHit:
    """검색 결과 단위 하나"""

    unit_id: str
    rank: int  # 1부터
    score: float
    best_part_id: str
    stage_ranks: dict[str, int] = field(default_factory=dict)  # bm25/dense/rrf/rerank (없으면 생략)


@dataclass
class RetrievalResult:
    """retrieve() 결과"""

    hits: list[UnitHit]
    timings_ms: dict[str, float]  # bm25, embed_query, dense, rrf, rerank, total
    n_rerank_truncated: int


def service_config(rerank: bool | None = None) -> RetrievalConfig:
    """서비스 경로 설정 — 프로필·재순위 백엔드·k값을 settings에서 읽는다"""
    return RetrievalConfig(
        profile=settings.service_index_profile,
        rerank=settings.rerank_enabled if rerank is None else rerank,
        reranker=settings.rerank_backend,
        k_bm25=settings.search_k_bm25,
        k_dense=settings.search_k_dense,
        rrf_k=settings.rrf_k,
        rerank_candidates=settings.rerank_candidates,
    )


_embedders: dict[str, EmbeddingService] = {}
_embedders_lock = threading.Lock()


def get_embedder(model: str) -> EmbeddingService:
    """모델별 질의 임베딩 서비스 (프로세스 내 재사용)"""
    with _embedders_lock:
        svc = _embedders.get(model)
        if svc is None:
            svc = _embedders[model] = EmbeddingService(model=model)
        return svc


def _rank_units(
    idx: CaseIndex,
    scores: np.ndarray,
    mask: np.ndarray | None,
    limit: int,
    positive_only: bool,
) -> list[tuple[str, int, float]]:
    """파트 점수 → 단위 순위 [(unit_id, 대표 파트 행, 점수)]. 단위별 첫 등장 파트가 대표.

    파트 동점은 행 번호 순으로 정해 결정적이다.
    """
    rows = np.arange(scores.shape[0])
    keep = np.ones(scores.shape[0], dtype=bool) if mask is None else mask.copy()
    if positive_only:
        keep &= scores > 0
    rows = rows[keep]
    if rows.size == 0:
        return []
    order = rows[np.lexsort((rows, -scores[rows]))]
    out: list[tuple[str, int, float]] = []
    seen: set[str] = set()
    for r in order:
        uid = idx.unit_ids[r]
        if uid in seen:
            continue
        seen.add(uid)
        out.append((uid, int(r), float(scores[r])))
        if len(out) >= limit:
            break
    return out


def _rrf(
    lists: dict[str, list[tuple[str, int, float]]], rrf_k: int
) -> list[tuple[str, int, float, dict[str, int]]]:
    """RRF 결합 → [(unit_id, 대표 행, 점수, {검색기: 순위})]"""
    ranks: dict[str, dict[str, int]] = {}
    rows: dict[str, dict[str, int]] = {}
    for name, lst in lists.items():
        for r, (uid, row, _s) in enumerate(lst, 1):
            ranks.setdefault(uid, {})[name] = r
            rows.setdefault(uid, {})[name] = row
    fused = []
    for uid, rk in ranks.items():
        # 합산 순서를 고정해 (a,b)·(b,a) 순위 쌍이 비트 단위로 같은 점수가 되게 한다
        score = sum(1.0 / (rrf_k + r) for r in sorted(rk.values()))
        best = min(rk, key=lambda n: (rk[n], n))
        fused.append((uid, rows[uid][best], score, rk))
    fused.sort(key=lambda x: (-x[2], min(x[3].values()), x[0]))
    return fused


def retrieve(
    case_id: str,
    query: str,
    cfg: RetrievalConfig | None = None,
    filters: dict[str, Any] | None = None,
    *,
    index: CaseIndex | None = None,
) -> RetrievalResult:
    """케이스 색인에서 단위 순위를 구한다 (서비스·실험 공용)

    Args:
        case_id: 케이스 ID
        query: 질의 원문
        cfg: 검색 설정 (기본 RetrievalConfig())
        filters: 서비스 필터 (CaseIndex.filter_mask 참고). 실험은 None
        index: 이미 로드한 색인 (없으면 case_index.load)

    Returns:
        RetrievalResult (hits는 rank 순, 최대 depth개)

    Raises:
        ValueError: BM25·Dense 둘 다 끈 설정
        IndexNotFoundError / IndexVersionError / EmbeddingDimensionError: 색인 문제
    """
    cfg = cfg or RetrievalConfig()
    if not (cfg.use_bm25 or cfg.use_dense):
        raise ValueError("use_bm25·use_dense 중 하나는 켜야 합니다")
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    idx = index or case_index.load(case_id)
    spec = profile_spec(cfg.profile)
    mask = idx.filter_mask(filters)
    single = cfg.use_bm25 != cfg.use_dense

    lists: dict[str, list[tuple[str, int, float]]] = {}
    if cfg.use_bm25:
        t0 = time.perf_counter()
        scores = idx.bm25_scores(query, spec.header)
        lists["bm25"] = _rank_units(idx, scores, mask, cfg.depth if single else cfg.k_bm25, True)
        timings["bm25"] = (time.perf_counter() - t0) * 1000
    if cfg.use_dense:
        embedder = get_embedder(spec.embed_model)
        idx.check_embedding(cfg.profile, spec.embed_model, embedder.get_dimension())
        t0 = time.perf_counter()
        qvec = embedder.embed_query(query)
        timings["embed_query"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        scores = idx.dense_scores(qvec, cfg.profile)
        lists["dense"] = _rank_units(idx, scores, mask, cfg.depth if single else cfg.k_dense, False)
        timings["dense"] = (time.perf_counter() - t0) * 1000

    # (unit_id, 대표 행, 점수, stage_ranks)
    ranked: list[tuple[str, int, float, dict[str, int]]]
    if single:
        name = "bm25" if cfg.use_bm25 else "dense"
        ranked = [(uid, row, s, {name: r}) for r, (uid, row, s) in enumerate(lists[name], 1)]
    else:
        t0 = time.perf_counter()
        ranked = []
        for r, (uid, row, s, rk) in enumerate(_rrf(lists, cfg.rrf_k), 1):
            ranked.append((uid, row, s, {**rk, "rrf": r}))
        timings["rrf"] = (time.perf_counter() - t0) * 1000
    ranked = ranked[: cfg.depth]

    n_truncated = 0
    if cfg.rerank and ranked:
        from src.rag.reranker import get_reranker

        t0 = time.perf_counter()
        head, tail = ranked[: cfg.rerank_candidates], ranked[cfg.rerank_candidates :]
        reranker = get_reranker(backend=cfg.reranker)
        before = reranker.n_truncated
        docs = [embed_input(idx.parts[row], spec.header) for _uid, row, _s, _rk in head]
        rr_scores = reranker.score(query, docs)
        n_truncated = reranker.n_truncated - before
        order = sorted(range(len(head)), key=lambda i: (-rr_scores[i], i))
        new_head = []
        for new_rank, i in enumerate(order, 1):
            uid, row, _s, rk = head[i]
            new_head.append((uid, row, float(rr_scores[i]), {**rk, "rerank": new_rank}))
        ranked = new_head + tail
        timings["rerank"] = (time.perf_counter() - t0) * 1000

    hits = [
        UnitHit(unit_id=uid, rank=r, score=s, best_part_id=idx.parts[row].part_id, stage_ranks=rk)
        for r, (uid, row, s, rk) in enumerate(ranked, 1)
    ]
    timings["total"] = (time.perf_counter() - t_start) * 1000
    return RetrievalResult(hits=hits, timings_ms=timings, n_rerank_truncated=n_truncated)
