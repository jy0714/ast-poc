"""실행: retrieve() 호출 → runs/{RUN}.trec, runs/{RUN}.timing.jsonl (지시서 §6.4, §6.7)

- 서비스와 같은 `src.rag.retrieval.retrieve()` 를 그대로 호출한다. 필터는 항상 None.
- V11: 검색 실험은 LLM 을 내린 상태에서 한다. LLM 이 적재돼 있으면 경고만 한다
  (`ollama stop <모델>` 로 내린 뒤 다시 실행 권장).
- 재순위 백엔드가 바뀔 때 이전 재순위 모델을 내려 VRAM 을 비운다.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ev4 import ollama
from ev4.io import ExperimentConfig, Query, write_jsonl, write_run
from src.index import case_index
from src.rag.retrieval import retrieve
from src.utils.config import settings


def run_all(
    cfg: ExperimentConfig,
    queries: list[Query],
    run_names: list[str],
    split: str,
    log: Callable[[str], None] = print,
) -> dict[str, float]:
    """실행 목록을 차례로 돌려 TREC·타이밍 파일을 쓴다

    Returns:
        {실행 이름: 걸린 초}
    """
    from src.rag.reranker import release_rerankers

    if not settings.ast_fake_models and ollama.is_loaded(settings.ollama_llm_model):
        log(f"[경고] LLM {settings.ollama_llm_model} 이 적재돼 있습니다. 검색 실험은 LLM 을 내린 상태에서 "
            f"합니다(V11): ollama stop {settings.ollama_llm_model}")
    idx = case_index.load(cfg.case_id)
    elapsed: dict[str, float] = {}
    last_reranker = None
    for name in run_names:
        rc = cfg.retrieval_config(name)
        if rc.rerank and last_reranker not in (None, rc.reranker):
            release_rerankers()
        if rc.rerank:
            last_reranker = rc.reranker
        t0 = time.perf_counter()
        run: dict[str, list[str]] = {}
        timing: list[dict] = []
        for q in queries:
            res = retrieve(cfg.case_id, q.text, rc, None, index=idx)
            run[q.qid] = [h.unit_id for h in res.hits]
            timing.append({
                "qid": q.qid,
                "n_hits": len(res.hits),
                "timings_ms": {k: round(v, 2) for k, v in res.timings_ms.items()},
                "n_rerank_truncated": res.n_rerank_truncated,
                "n_reranked": min(rc.rerank_candidates, len(res.hits)) if rc.rerank else 0,
            })
        write_run(cfg.run_path(name, split), name, run, cfg.depth)
        write_jsonl(cfg.timing_path(name, split), timing)
        elapsed[name] = time.perf_counter() - t0
        empty = sum(1 for v in run.values() if not v)
        log(f"[run] {name} ({split}) 질의 {len(queries)}건 · {elapsed[name]:.1f}s"
            + (f" · 결과 없는 질의 {empty}건" if empty else ""))
    return elapsed
