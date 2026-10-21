"""응답 시간·VRAM·응답 생성 → results/T7_latency.md, runs/gen/answers.jsonl (지시서 §6.7, v4.4 §4·§5)

- 서비스 조건: 측정 전 세 모델(임베딩·재순위·LLM)을 모두 적재한다.
- 검색: latency_runs(v4.4 §5: 논문 R5 = 하네스 R8 하나) × 질문 45개. 사전 실행 latency_warmup 개 후,
  질의마다 latency_repeats 회 반복(실행이 여럿이면 순서 무작위). 단계별(timings_ms) p50/p95, 재순위 절단 비율.
- 생성: 생성 기준 실행(기본 R8) 상위 5개 단위를 컨텍스트로 LLM 응답(질의당 1회). 컨텍스트는 서비스와 같은
  함수(`src.rag.engine.format_part_context` = 대표 파트 → `_format_source_context`)로 만든다(v4.4 §4.1).
  답변·컨텍스트 단위 ID·인용 번호(서비스 인용 정규식)·인용 → 단위 매핑·모델 태그를
  runs/gen/answers.jsonl 에 남긴다(P0 — 답변 본문은 원문에 해당). 이 답변이 G1 판정의 입력이다.
- v4.4 §5 보고 값: 검색 전체 p50/p95, 재순위 p50, 응답 생성 포함 질의당 전체 시간 p50, GPU 최대 사용량
  (nvidia-smi·torch·ollama ps), 재순위 입력 절단 비율(settings.rerank_max_length 초과).
- VRAM: 측정 동안 nvidia-smi memory.used 최대치(0.5초 간격), 재순위 프로세스 torch 최대 할당량.
  서비스 세 모델만 올린 시점의 값을 따로 기록한다(v4.2 L3).
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
from ev4.env import ollama_models
from ev4.io import ExperimentConfig, Query, write_jsonl
from ev4.stats import percentile
from src.index import case_index
from src.rag.retrieval import retrieve
from src.utils.config import settings

GEN_TOP_K = 5
STAGES = ("bm25", "embed_query", "dense", "rrf", "rerank", "total")
# 논문 [표 13] 단계 이름 (표시 순서 고정, v4.2 L1)
STAGE_LABELS = (
    ("embed_query", "질의 임베딩"),
    ("dense", "밀집 검색"),
    ("bm25", "BM25 검색"),
    ("rrf", "RRF 결합"),
    ("rerank", "재순위"),
    ("total", "검색 전체"),
)


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
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
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


async def _fake_stream(prompt: str, n_context: int) -> AsyncIterator[str]:
    """가짜 응답 — 컨텍스트가 있으면 인용 두 개, 없으면 거절 문구"""
    toks = (
        ("가짜 응답입니다 [출처 1].", " 두 번째 문장입니다 [출처 2].")
        if n_context
        else ("자료에서 해당 내용을 찾을 수 없습니다.",)
    )
    for tok in toks:
        await asyncio.sleep(0)
        yield tok
    del prompt


async def _measure_generation(question: str, context: list[str]) -> dict[str, Any]:
    """첫 토큰까지·전체 시간·출력 토큰 수·답변 전문"""
    usage: dict[str, Any] = {}
    t0 = time.perf_counter()
    first: float | None = None
    n_chunks = 0
    chunks: list[str] = []
    if settings.ast_fake_models:
        stream = _fake_stream(question, len(context))
    else:
        from src.llm.router import LLMRouter

        stream = LLMRouter().generate_stream(
            question, context, secure_mode=True, on_complete=usage.update
        )
    async for tok in stream:
        if first is None:
            first = time.perf_counter()
        n_chunks += 1
        chunks.append(tok)
    total = time.perf_counter() - t0
    out_tokens = int(usage.get("output_tokens") or n_chunks)
    gen_time = total - ((first or t0) - t0)
    return {
        "ttft_ms": ((first or t0) - t0) * 1000,
        "total_ms": total * 1000,
        "output_tokens": out_tokens,
        "tokens_per_s": out_tokens / gen_time if gen_time > 0 else float("nan"),
        "token_count_source": "usage" if usage.get("output_tokens") else "chunks",
        "answer": "".join(chunks),
    }


def _model_rows(loaded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`ollama ps` 행에서 메모리 관련 값만 (상주 크기·VRAM 상주·컨텍스트 길이)"""
    return [
        {
            "name": m.get("name"),
            "size": m.get("size"),
            "size_vram": m.get("size_vram"),
            "context_length": m.get("context_length"),
        }
        for m in loaded
    ]


def _memory_snapshot(mon: VramMonitor) -> dict[str, Any]:
    """현재까지의 nvidia-smi 최대·현재 사용량, torch 최대 할당, ollama ps"""
    return {
        "vram_max_mib": mon.max_mib,
        "vram_now_mib": None if settings.ast_fake_models else mon.sample(),
        "torch_peak_mib": torch_peak_mib(),
        "loaded_models": _model_rows(ollama.loaded_models()),
    }


def _context(idx: case_index.CaseIndex, hits: list[Any]) -> tuple[list[str], list[str]]:
    """상위 GEN_TOP_K 단위 → (단위 ID, 서비스와 같은 컨텍스트 문자열) — 대표 파트(best_part_id) 기준"""
    from src.rag.engine import format_part_context

    uids: list[str] = []
    texts: list[str] = []
    for h in hits[:GEN_TOP_K]:
        uids.append(h.unit_id)
        texts.append(format_part_context(idx.part(h.best_part_id), h.score))
    return uids, texts


def answer_record(
    qid: str, run: str, answer: str, unit_ids: list[str], model: str
) -> dict[str, Any]:
    """답변 기록 한 줄 (v4.4 §4.1) — 인용 번호는 서비스 정규식, 인용 → 단위 ID 매핑"""
    from src.rag.engine import extract_citations, is_refusal, validate_citations

    cites = extract_citations(answer)
    count, invalid, uncited = validate_citations(answer, len(unit_ids))
    return {
        "qid": qid,
        "run": run,
        "model": model,
        "answer": answer,
        "context_unit_ids": unit_ids,
        "citations": cites,
        "citation_map": {
            str(n): unit_ids[n - 1] for n in sorted(set(cites)) if 1 <= n <= len(unit_ids)
        },
        "citation_count": count,
        "invalid_citations": invalid,
        "uncited_response": uncited,
        "refusal": is_refusal(answer),
    }


# ─────────────────────────── 측정 ───────────────────────────


def _q(values: list[float]) -> dict[str, float]:
    return {"p50": percentile(values, 50), "p95": percentile(values, 95), "n": len(values)}


def measure(
    cfg: ExperimentConfig, queries: list[Query], log: Callable[[str], None] = print
) -> dict[str, Any]:
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
        # 1) 서비스 세 모델 적재(생성 기준 실행의 임베딩·재순위 + LLM) → 메모리 기록 (v4.2 L3)
        retrieve(cfg.case_id, queries[0].text, gen_rc, None, index=idx)
        asyncio.run(_measure_generation(queries[0].text, []))
        service_mem = _memory_snapshot(mon)
        # 2) 나머지 실행 적재 — 실행마다 한 번 검색
        for r in runs:
            retrieve(cfg.case_id, queries[0].text, rcs[r], None, index=idx)
        loaded = ollama.loaded_models()
        # 3) 사전 실행
        for q in queries[: cfg.latency_warmup]:
            for r in runs:
                retrieve(cfg.case_id, q.text, rcs[r], None, index=idx)
        # 4) 본 측정
        rows: list[dict[str, Any]] = []
        for rep in range(cfg.latency_repeats):
            for q in queries:
                order = runs[:]
                rng.shuffle(order)
                for r in order:
                    res = retrieve(cfg.case_id, q.text, rcs[r], None, index=idx)
                    rows.append(
                        {
                            "run": r,
                            "qid": q.qid,
                            "rep": rep,
                            "timings_ms": res.timings_ms,
                            "n_rerank_truncated": res.n_rerank_truncated,
                            "n_reranked": min(rcs[r].rerank_candidates, len(res.hits))
                            if rcs[r].rerank
                            else 0,
                        }
                    )
        log(f"[latency] 검색 {len(rows)}회 측정")
        # 5) 생성 (질의당 1회, 이 답변이 G1 판정 입력)
        model = "fake" if settings.ast_fake_models else settings.ollama_llm_model
        gens: list[dict[str, Any]] = []
        answers: list[dict[str, Any]] = []
        for q in queries:
            res = retrieve(cfg.case_id, q.text, gen_rc, None, index=idx)
            uids, ctx = _context(idx, res.hits)
            g = asyncio.run(_measure_generation(q.text, ctx))
            answer = g.pop("answer")
            retrieval_ms = float(res.timings_ms.get("total", 0.0))
            gens.append(
                {
                    "qid": q.qid,
                    "run": gen_run,
                    "n_context": len(ctx),
                    "retrieval_ms": retrieval_ms,
                    "query_total_ms": retrieval_ms + g["total_ms"],
                    **g,
                }
            )
            answers.append(answer_record(q.qid, gen_run, answer, uids, model))
        log(f"[latency] 생성 {len(gens)}회 측정 · 답변 기록 {len(answers)}건")
    vram_max = mon.max_mib
    torch_peak = torch_peak_mib()

    write_jsonl(cfg.runs_dir / "latency.jsonl", rows)
    write_jsonl(cfg.runs_dir / "generation.jsonl", gens)
    write_jsonl(cfg.runs_dir / "gen" / "answers.jsonl", answers)

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
        "generation": {
            k: _q([g[k] for g in gens])
            for k in ("ttft_ms", "total_ms", "tokens_per_s", "query_total_ms")
        },
        "generation_run": gen_run,
        "answers": {
            "n": len(answers),
            "with_citation": sum(1 for a in answers if a["citation_count"]),
            "refusal": sum(1 for a in answers if a["refusal"]),
            "invalid_citation": sum(1 for a in answers if a["invalid_citations"]),
        },
        "vram_max_mib": vram_max,
        "torch_peak_mib": torch_peak,
        "loaded_models": _model_rows(loaded),
        "service_memory": service_mem,
        "models": ollama_models([settings.ollama_embed_model, settings.ollama_llm_model]),
        "llm_num_ctx": settings.llm_num_ctx,
        "rerank_max_length": settings.rerank_max_length,
        "reranker": {r: rcs[r].reranker for r in runs if rcs[r].rerank},
        "labels": {r: cfg.label(r) for r in {*runs, gen_run}},
        "n_queries": len(queries),
        "warmup": cfg.latency_warmup,
        "repeats": cfg.latency_repeats,
        "fake_models": settings.ast_fake_models,
    }
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "latency.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (cfg.results_dir / "T7_latency.md").write_text(render_t7(summary), encoding="utf-8")
    log(f"[latency] → {cfg.results_dir / 'T7_latency.md'}")
    return summary


def _ms(d: dict[str, float] | None) -> str:
    if not d:
        return "–"
    return f"{d['p50']:,.1f} / {d['p95']:,.1f}"


def _mib(v: float | None) -> str:
    return "–" if v is None else f"{v:,.0f}"


def _bytes_mib(v: Any) -> str:
    return f"{int(v) / 2**20:,.0f}" if v else "–"


def v44_values_lines(s: dict[str, Any]) -> list[str]:
    """v4.4 §5 보고 값 (논문 4.3절 본문 두 문단) — 표 대신 값"""
    labels = s.get("labels", {})
    gen_run = s["generation_run"]
    run = gen_run if gen_run in s["runs"] else next(iter(s["runs"]))
    st = s["runs"][run]
    svc = s.get("service_memory") or {}
    g = s["generation"]
    tr = s["rerank_truncation"].get(run)
    max_len = s.get("rerank_max_length", settings.rerank_max_length)
    models = ", ".join(
        f"{m.get('name')} {_bytes_mib(m.get('size_vram'))} MiB"
        for m in svc.get("loaded_models") or []
    )

    def _p(key: str, q: str) -> str:
        d = st.get(key)
        return "–" if not d else f"{d[q]:,.1f}"

    return [
        f"## 응답 시간·GPU 메모리 보고 값 (v4.4 §5) — {labels.get(run, run)} (하네스 {run})",
        "",
        f"- 검색 전체: p50 {_p('total', 'p50')} ms · p95 {_p('total', 'p95')} ms",
        f"- 그중 재순위: p50 {_p('rerank', 'p50')} ms",
        "- 응답 생성 포함 질의당 전체 시간: "
        + (
            f"p50 {qt['p50']:,.1f} ms (질문 {qt['n']}개, 질의당 생성 1회)"
            if (qt := g.get("query_total_ms"))
            else "–"
        ),
        f"- GPU 최대 사용량 (세 모델 적재): nvidia-smi {_mib(svc.get('vram_max_mib'))} MiB "
        f"(측정 전체 {_mib(s['vram_max_mib'])} MiB) · 재순위 torch {_mib(s['torch_peak_mib'])} MiB · "
        f"ollama ps: {models or '–'}",
        f"- 재순위 입력 {max_len:,}토큰 초과 절단 비율: {'–' if tr is None else f'{tr:.1%}'}",
        "",
    ]


def render_t7(s: dict[str, Any]) -> str:
    """T7 마크다운 — v4.4 §5 보고 값, 단계별 시간(참고), 생성, 재순위 절단, 모델별 메모리"""
    runs = list(s["runs"])
    labels = s.get("labels", {})
    lines = [
        "# T7 응답 시간·메모리·생성 시간",
        "",
        f"- 평가 질의 {s['n_queries']}건 × 반복 {s['repeats']}회 (사전 실행 {s['warmup']}건)"
        + (" · **가짜 모델**" if s["fake_models"] else ""),
        "- 세 모델(임베딩·재순위·LLM) 동시 적재 상태",
        "",
        *v44_values_lines(s),
        "## 검색 단계별 시간 (ms, p50 / p95, 참고)",
        "",
        "| 단계 | " + " | ".join(labels.get(r, r) for r in runs) + " |",
        "|---|" + "---:|" * len(runs),
    ]
    for key, label in STAGE_LABELS:
        lines.append(f"| {label} | " + " | ".join(_ms(s["runs"][r].get(key)) for r in runs) + " |")
    max_len = s.get("rerank_max_length", settings.rerank_max_length)
    lines += [
        "",
        f"## 재순위 입력 절단 ({max_len:,}토큰 초과 후보 비율)",
        "",
        "| 실행 | 절단 비율 |",
        "|---|---:|",
    ]
    for r in runs:
        tr = s["rerank_truncation"].get(r)
        lines.append(f"| {labels.get(r, r)} | {'–' if tr is None else f'{tr:.1%}'} |")
    g = s["generation"]
    lines += [
        "",
        f"## 응답 생성 ({labels.get(s['generation_run'], s['generation_run'])} 상위 {GEN_TOP_K}개 단위 컨텍스트, 질의당 1회)",
        "",
        "| 항목 | p50 | p95 |",
        "|---|---:|---:|",
        f"| 첫 토큰까지 (ms) | {g['ttft_ms']['p50']:,.1f} | {g['ttft_ms']['p95']:,.1f} |",
        f"| 전체 생성 (ms) | {g['total_ms']['p50']:,.1f} | {g['total_ms']['p95']:,.1f} |",
        f"| 초당 생성 토큰 | {g['tokens_per_s']['p50']:,.1f} | {g['tokens_per_s']['p95']:,.1f} |",
    ]
    svc = s.get("service_memory") or {}
    lines += [
        "",
        "## GPU 메모리 (v4.2 L3)",
        "",
        "| 항목 | MiB |",
        "|---|---:|",
        f"| nvidia-smi 최대 — 서비스 세 모델 적재 시점까지 | {_mib(svc.get('vram_max_mib'))} |",
        f"| nvidia-smi 현재 — 서비스 세 모델 적재 직후 | {_mib(svc.get('vram_now_mib'))} |",
        f"| nvidia-smi 최대 — 측정 전체(비교 실행 모델 포함) | {_mib(s['vram_max_mib'])} |",
        f"| 재순위 torch 최대 할당 — 서비스 재순위기만 | {_mib(svc.get('torch_peak_mib'))} |",
        f"| 재순위 torch 최대 할당 — 측정 전체 | {_mib(s['torch_peak_mib'])} |",
        "",
        f"- LLM 컨텍스트 길이 설정(LLM_NUM_CTX, KV 캐시 크기 결정): {s.get('llm_num_ctx', '–')}",
        "",
        "| Ollama 모델 (서비스 적재 시점 ollama ps) | 상주 크기 (MiB) | VRAM 상주 (MiB) | 컨텍스트 길이 |",
        "|---|---:|---:|---:|",
    ]
    for m in svc.get("loaded_models") or []:
        lines.append(
            f"| {m.get('name')} | {_bytes_mib(m.get('size'))} | {_bytes_mib(m.get('size_vram'))} | "
            f"{m.get('context_length') or '–'} |"
        )
    lines += [
        "",
        "| 모델 (ollama show) | 양자화 | 매개변수 | 파일 크기 (MiB) | digest |",
        "|---|---|---|---:|---|",
    ]
    for m, info in s["models"].items():
        lines.append(
            f"| {m} | {info.get('quantization') or '–'} | {info.get('parameter_size') or '–'} | "
            f"{_bytes_mib(info.get('file_size_bytes'))} | {str(info.get('digest', ''))[:12] or '–'} |"
        )
    lines += [
        "",
        "- 측정 종료 시점 적재 모델(ollama ps): "
        + (", ".join(f"{m['name']}" for m in s["loaded_models"]) or "–"),
    ]
    return "\n".join(lines) + "\n"
