"""응답 시간·VRAM·생성 시간 → results/T7_latency.md (지시서 §6.7, §6.8 T7)

- 서비스 조건: 측정 전 세 모델(임베딩·재순위·LLM)을 모두 적재한다.
- 검색: latency_runs × 평가 질의 전체. 사전 실행 latency_warmup 개 후, 질의마다 실행 순서를
  무작위로 바꿔 latency_repeats 회 반복. 단계별(timings_ms) p50/p95, 재순위 절단 비율.
- 생성: 생성 기준 실행(기본 R8) 상위 5개 단위를 컨텍스트로 LLM 응답(1회). 첫 토큰까지·전체 시간·초당 토큰.
- VRAM: 측정 동안 nvidia-smi memory.used 최대치(0.5초 간격), 재순위 프로세스 torch 최대 할당량.
- AST_FAKE_MODELS=1 이면 LLM 은 가짜 스트림, GPU 기록은 빈 값.
"""

from __future__ import annotations

import asyncio
import json
import random
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from typing import Any

from ev4 import ollama
from ev4.io import ExperimentConfig, Query, write_jsonl
from ev4.stats import percentile
from src.index import case_index
from src.rag.retrieval import retrieve
from src.utils.config import settings

GEN_TOP_K = 5
GEN_UNIT_MAX_CHARS = 4000
STAGES = ("bm25", "embed_query", "dense", "rrf", "rerank", "total")


class VramMonitor:
    """nvidia-smi memory.used 최대치를 백그라운드에서 기록 (없으면 None)"""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.max_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def sample() -> int | None:
        """현재 GPU 0 사용 메모리(MiB)"""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout
            return int(out.strip().splitlines()[0])
        except Exception:
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self.sample()
            if v is None:
                return
            self.max_mib = max(self.max_mib or 0, v)
            self._stop.wait(self.interval)

    def __enter__(self) -> VramMonitor:
        if not settings.ast_fake_models:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)


def torch_peak_mib() -> float | None:
    """이 프로세스의 torch.cuda.max_memory_allocated (MiB)"""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**20
    except Exception:
        pass
    return None


def _reset_torch_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ─────────────────────────── 생성 ───────────────────────────


async def _fake_stream(prompt: str) -> AsyncIterator[str]:
    for tok in ("가짜", " 응답", "입니다", "."):
        await asyncio.sleep(0)
        yield tok
    del prompt


async def _measure_generation(question: str, context: list[str]) -> dict[str, Any]:
    """첫 토큰까지·전체 시간·출력 토큰 수"""
    usage: dict[str, Any] = {}
    t0 = time.perf_counter()
    first: float | None = None
    n_chunks = 0
    if settings.ast_fake_models:
        stream = _fake_stream(question)
    else:
        from src.llm.router import LLMRouter

        stream = LLMRouter().generate_stream(question, context, secure_mode=True, on_complete=usage.update)
    async for _tok in stream:
        if first is None:
            first = time.perf_counter()
        n_chunks += 1
    total = time.perf_counter() - t0
    out_tokens = int(usage.get("output_tokens") or n_chunks)
    gen_time = total - ((first or t0) - t0)
    return {
        "ttft_ms": ((first or t0) - t0) * 1000,
        "total_ms": total * 1000,
        "output_tokens": out_tokens,
        "tokens_per_s": out_tokens / gen_time if gen_time > 0 else float("nan"),
        "token_count_source": "usage" if usage.get("output_tokens") else "chunks",
    }


def _context(idx: case_index.CaseIndex, unit_ids: list[str]) -> list[str]:
    out = []
    for uid in unit_ids[:GEN_TOP_K]:
        parts = idx.unit_parts(uid)
        if parts:
            out.append(f"{parts[0].header}\n\n{idx.unit_text(uid)[:GEN_UNIT_MAX_CHARS]}")
    return out


# ─────────────────────────── 측정 ───────────────────────────


def _q(values: list[float]) -> dict[str, float]:
    return {"p50": percentile(values, 50), "p95": percentile(values, 95), "n": len(values)}


def measure(cfg: ExperimentConfig, queries: list[Query], log: Callable[[str], None] = print) -> dict[str, Any]:
    """T7 측정 → runs/latency.jsonl, runs/generation.jsonl, results/latency.json, results/T7_latency.md"""
    idx = case_index.load(cfg.case_id)
    runs = list(cfg.latency_runs)
    gen_run = str(cfg.extra.get("generation_run", "R8"))
    if not runs or not queries:
        raise ValueError("latency_runs 와 test 질의가 필요합니다")
    rcs = {r: cfg.retrieval_config(r) for r in runs}
    gen_rc = rcs.get(gen_run) or cfg.retrieval_config(gen_run)
    rng = random.Random(cfg.seed)
    _reset_torch_peak()

    with VramMonitor() as mon:
        # 1) 세 모델 적재 — 실행마다 한 번 검색(임베딩·재순위), LLM 은 짧은 생성 한 번
        for r in runs:
            retrieve(cfg.case_id, queries[0].text, rcs[r], None, index=idx)
        asyncio.run(_measure_generation(queries[0].text, _context(idx, [])))
        loaded = ollama.loaded_models()
        # 2) 사전 실행
        for q in queries[: cfg.latency_warmup]:
            for r in runs:
                retrieve(cfg.case_id, q.text, rcs[r], None, index=idx)
        # 3) 본 측정
        rows: list[dict[str, Any]] = []
        for rep in range(cfg.latency_repeats):
            for q in queries:
                order = runs[:]
                rng.shuffle(order)
                for r in order:
                    res = retrieve(cfg.case_id, q.text, rcs[r], None, index=idx)
                    rows.append({"run": r, "qid": q.qid, "rep": rep, "timings_ms": res.timings_ms,
                                 "n_rerank_truncated": res.n_rerank_truncated,
                                 "n_reranked": min(rcs[r].rerank_candidates, len(res.hits)) if rcs[r].rerank else 0})
        log(f"[latency] 검색 {len(rows)}회 측정")
        # 4) 생성
        gens: list[dict[str, Any]] = []
        for q in queries:
            res = retrieve(cfg.case_id, q.text, gen_rc, None, index=idx)
            ctx = _context(idx, [h.unit_id for h in res.hits])
            g = asyncio.run(_measure_generation(q.text, ctx))
            gens.append({"qid": q.qid, "run": gen_run, "n_context": len(ctx), **g})
        log(f"[latency] 생성 {len(gens)}회 측정")
    vram_max = mon.max_mib
    torch_peak = torch_peak_mib()

    write_jsonl(cfg.runs_dir / "latency.jsonl", rows)
    write_jsonl(cfg.runs_dir / "generation.jsonl", gens)

    by_run: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    trunc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        for k, v in row["timings_ms"].items():
            by_run[row["run"]][k].append(v)
        trunc[row["run"]][0] += row["n_rerank_truncated"]
        trunc[row["run"]][1] += row["n_reranked"]
    summary = {
        "runs": {r: {s: _q(by_run[r][s]) for s in STAGES if by_run[r].get(s)} for r in runs},
        "rerank_truncation": {r: (t[0] / t[1] if t[1] else None) for r, t in trunc.items()},
        "generation": {k: _q([g[k] for g in gens]) for k in ("ttft_ms", "total_ms", "tokens_per_s")},
        "generation_run": gen_run,
        "vram_max_mib": vram_max,
        "torch_peak_mib": torch_peak,
        "loaded_models": [{"name": m.get("name"), "size_vram": m.get("size_vram")} for m in loaded],
        "models": {m: ollama.show(m) for m in (settings.ollama_embed_model, settings.ollama_llm_model)},
        "reranker": {r: rcs[r].reranker for r in runs if rcs[r].rerank},
        "n_queries": len(queries), "warmup": cfg.latency_warmup, "repeats": cfg.latency_repeats,
        "fake_models": settings.ast_fake_models,
    }
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "latency.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (cfg.results_dir / "T7_latency.md").write_text(render_t7(summary), encoding="utf-8")
    log(f"[latency] → {cfg.results_dir / 'T7_latency.md'}")
    return summary


def _ms(d: dict[str, float] | None) -> str:
    if not d:
        return "–"
    return f"{d['p50']:,.1f} / {d['p95']:,.1f}"


def render_t7(s: dict[str, Any]) -> str:
    """T7 마크다운"""
    lines = [
        "# T7 응답 시간·VRAM·생성 시간",
        "",
        f"- 평가 질의 {s['n_queries']}건 × 반복 {s['repeats']}회 (사전 실행 {s['warmup']}건, 질의마다 실행 순서 무작위)"
        + (" · **가짜 모델**" if s["fake_models"] else ""),
        "- 세 모델(임베딩·재순위·LLM) 동시 적재 상태",
        "",
        "## 검색 단계별 시간 (ms, p50 / p95)",
        "",
        "| 실행 | " + " | ".join(STAGES) + " | 재순위 절단 비율 |",
        "|---|" + "---:|" * (len(STAGES) + 1),
    ]
    for r, st in s["runs"].items():
        tr = s["rerank_truncation"].get(r)
        tr_s = "–" if tr is None else f"{tr:.1%}"
        lines.append(f"| {r} | " + " | ".join(_ms(st.get(k)) for k in STAGES) + f" | {tr_s} |")
    g = s["generation"]
    vram = "–" if s["vram_max_mib"] is None else f"{s['vram_max_mib']:,}"
    peak = "–" if s["torch_peak_mib"] is None else f"{s['torch_peak_mib']:,.0f}"
    lines += [
        "",
        f"## 생성 ({s['generation_run']} 상위 {GEN_TOP_K}개 단위 컨텍스트, 1회)",
        "",
        "| 항목 | p50 | p95 |",
        "|---|---:|---:|",
        f"| 첫 토큰까지 (ms) | {g['ttft_ms']['p50']:,.1f} | {g['ttft_ms']['p95']:,.1f} |",
        f"| 전체 (ms) | {g['total_ms']['p50']:,.1f} | {g['total_ms']['p95']:,.1f} |",
        f"| 초당 토큰 | {g['tokens_per_s']['p50']:,.1f} | {g['tokens_per_s']['p95']:,.1f} |",
        "",
        "## VRAM",
        "",
        f"- nvidia-smi memory.used 최대: {vram} MiB",
        f"- 재순위 torch 최대 할당: {peak} MiB",
        "- 적재 모델(ollama ps): " + (", ".join(f"{m['name']}" for m in s["loaded_models"]) or "–"),
    ]
    for m, info in s["models"].items():
        lines.append(f"- {m}: 양자화 {info.get('quantization', '–')} · 크기 {info.get('parameter_size', '–')} · "
                     f"digest {str(info.get('digest', ''))[:12] or '–'}")
    return "\n".join(lines) + "\n"
