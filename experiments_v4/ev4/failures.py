"""실패 분석 시트 → results/F1_failures.csv, results/F1_summary.md (지시서 §6.7, §6.8 F1)

- 대상: 기준 실행(기본 R8)에서 등급 2 단위가 상위 10 에 없는 질의.
- 행: 놓친 등급 2 단위마다 한 행. R2·R4·R6·R7·R8 순위(100위 밖이면 ">100").
- `category`·`note` 는 사용자가 채운다. 다시 실행하면 같은 (qid, unit_id) 의 기존 분류를 보존한다.
- 요약: 분류별 건수(채워진 것만)와 질의 유형별 실패 질의 수.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ev4.io import ExperimentConfig, read_qrels, read_queries, read_run
from src.index import case_index

CATEGORIES = ("토큰화(정확 일치 실패)", "교차 언어", "표현 불일치", "긴 단위 희석", "재순위 강등",
              "헤더 영향", "판정 이견", "기타")
DEFAULT_RANK_RUNS = ["R2", "R4", "R6", "R7", "R8"]
CUTOFF = 10


def _read_existing(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {(r["qid"], r["unit_id"]): (r.get("category", ""), r.get("note", "")) for r in csv.DictReader(f)}


def build_failures(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """실패 분석 시트·요약 생성

    Returns:
        {n_queries_failed, n_rows, categories}
    """
    base = str(cfg.extra.get("failure_run", "R8"))
    rank_runs = [r for r in cfg.extra.get("failure_rank_runs", DEFAULT_RANK_RUNS) if r in cfg.runs]
    if base not in rank_runs:
        rank_runs.append(base)
    queries = {q.qid: q for q in read_queries(cfg.queries_path, split="test")}
    qrels = read_qrels(cfg.qrels_path)
    runs = {r: read_run(cfg.run_path(r, "test")) for r in rank_runs if cfg.run_path(r, "test").exists()}
    if base not in runs:
        raise FileNotFoundError(f"기준 실행 {base} 파일이 없습니다: {cfg.run_path(base, 'test')}")
    idx = case_index.load(cfg.case_id)
    out = cfg.results_dir / "F1_failures.csv"
    prev = _read_existing(out)

    rows: list[dict[str, Any]] = []
    failed_q: list[str] = []
    for qid in sorted(qrels):
        if qid not in queries:
            continue
        top = set(runs[base].get(qid, [])[:CUTOFF])
        missed = sorted(u for u, g in qrels[qid].items() if g == 2 and u not in top)
        if not missed:
            continue
        failed_q.append(qid)
        q = queries[qid]
        for uid in missed:
            parts = idx.unit_parts(uid)
            meta = parts[0].meta if parts else {}
            row: dict[str, Any] = {
                "qid": qid, "type": q.type, "query": q.text, "unit_id": uid,
                "unit_type": parts[0].unit_type if parts else "", "lang": meta.get("lang", ""),
                "n_tokens": sum(p.n_tokens for p in parts),
            }
            for r in rank_runs:
                ranked = runs.get(r, {}).get(qid, [])
                row[f"rank_{r}"] = ranked.index(uid) + 1 if uid in ranked[: cfg.depth] else f">{cfg.depth}"
            cat, note = prev.get((qid, uid), ("", ""))
            row["category"], row["note"] = cat, note
            rows.append(row)

    cols = ["qid", "type", "query", "unit_id", "unit_type", "lang", "n_tokens",
            *[f"rank_{r}" for r in rank_runs], "category", "note"]
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    cats = Counter(r["category"] for r in rows if r["category"])
    by_type = Counter(queries[q].type for q in failed_q)
    md = [
        f"# F1 실패 분석 — 기준 {base}, 등급 2 단위가 상위 {CUTOFF} 밖",
        "",
        f"- 실패 질의 {len(failed_q)}건 · 놓친 단위 {len(rows)}건 · 분류 완료 {sum(cats.values())}건",
        "- 질의 유형별 실패 질의: " + (", ".join(f"{t} {n}" for t, n in sorted(by_type.items())) or "–"),
        "",
        "| 분류 | 건수 |",
        "|---|---:|",
        *[f"| {c} | {cats.get(c, 0)} |" for c in CATEGORIES],
    ]
    unknown = {c: n for c, n in cats.items() if c not in CATEGORIES}
    md += [f"| {c} (목록 외) | {n} |" for c, n in sorted(unknown.items())]
    md += ["", f"분류 선택지: {' / '.join(CATEGORIES)} — `F1_failures.csv` 의 category 칸에 입력"]
    (cfg.results_dir / "F1_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"[failures] 실패 질의 {len(failed_q)} · 행 {len(rows)} → {out}")
    return {"n_queries_failed": len(failed_q), "n_rows": len(rows), "categories": dict(cats)}
