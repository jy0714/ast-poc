"""실데이터 동작 점검 → results/smoke.md (v4.3 §2, 개발용 질의 대체)

표준 질문 3개(config `smoke_qids`, 기본 q1.1·q2.6·q7.2)를 점검 실행(config `smoke_runs`, 기본 R2·R4·R6·R8·R9)으로
검색해 오류 여부, 응답 시간, GPU 메모리만 기록한다. 순위·점수·실행 파일은 남기지 않는다.
이 결과를 보고 설정을 바꾸지 않는다(설정은 v4.1 기본값 고정).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ev4.io import ExperimentConfig, Query
from ev4.latency import VramMonitor
from src.index import case_index
from src.rag.retrieval import retrieve

DEFAULT_SMOKE_RUNS = ["R2", "R4", "R6", "R8", "R9"]


def smoke(cfg: ExperimentConfig, queries: list[Query], log: Callable[[str], None] = print) -> dict[str, Any]:
    """점검 질의 × 점검 실행 검색 → 오류 수·응답 시간·GPU 메모리

    Returns:
        {qids, runs: {run: {n, errors, error_types, mean_ms, max_ms}}, vram_max_mib}

    Raises:
        ValueError: smoke_qids 가 질의 파일에 없을 때
    """
    by_id = {q.qid: q for q in queries}
    missing = [q for q in cfg.smoke_qids if q not in by_id]
    if missing:
        raise ValueError(f"smoke_qids 가 질의 파일에 없습니다: {missing}")
    run_names = [str(r) for r in cfg.extra.get("smoke_runs", DEFAULT_SMOKE_RUNS)]
    unknown = [r for r in run_names if r not in cfg.runs]
    if unknown:
        raise ValueError(f"config 에 없는 점검 실행: {unknown}")
    idx = case_index.load(cfg.case_id)
    out: dict[str, Any] = {"qids": list(cfg.smoke_qids), "runs": {}}
    with VramMonitor() as mon:
        for name in run_names:
            rc = cfg.retrieval_config(name)
            times: list[float] = []
            errors: list[str] = []
            for qid in cfg.smoke_qids:
                t0 = time.perf_counter()
                try:
                    retrieve(cfg.case_id, by_id[qid].text, rc, None, index=idx)
                except Exception as e:  # 점검은 오류 유형만 기록하고 계속한다
                    errors.append(type(e).__name__)
                    continue
                times.append((time.perf_counter() - t0) * 1000)
            out["runs"][name] = {
                "n": len(cfg.smoke_qids), "errors": len(errors), "error_types": sorted(set(errors)),
                "mean_ms": sum(times) / len(times) if times else None, "max_ms": max(times, default=None),
            }
            log(f"[smoke] {name}: 질의 {len(cfg.smoke_qids)}건 · 오류 {len(errors)}건")
    out["vram_max_mib"] = mon.max_mib
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "smoke.md").write_text(render_smoke(out), encoding="utf-8")
    log(f"[smoke] → {cfg.results_dir / 'smoke.md'}")
    return out


def _ms(v: float | None) -> str:
    return "–" if v is None else f"{v:,.0f}"


def render_smoke(s: dict[str, Any]) -> str:
    """점검 표 (오류·응답 시간·GPU 메모리만 — 순위·점수 없음)"""
    lines = ["# 실데이터 동작 점검 (v4.3 §2)", "",
             f"- 점검 질의: {', '.join(s['qids'])} · 첫 질의는 모델 적재 시간을 포함한다",
             "- 이 결과로 설정을 바꾸지 않는다", "",
             "| 실행 | 질의 | 오류 | 오류 유형 | 응답 시간 평균(ms) | 최대(ms) |", "|---|---:|---:|---|---:|---:|"]
    for r, v in s["runs"].items():
        lines.append(f"| {r} | {v['n']} | {v['errors']} | {', '.join(v['error_types']) or '–'} | "
                     f"{_ms(v['mean_ms'])} | {_ms(v['max_ms'])} |")
    vram = s.get("vram_max_mib")
    lines += ["", f"GPU 메모리 최대(nvidia-smi): {f'{vram:,} MiB' if vram else '–'}"]
    return "\n".join(lines) + "\n"
