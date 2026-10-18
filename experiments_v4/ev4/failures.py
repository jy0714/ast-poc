"""실패 분석 시트 → results/F1_failures.csv, results/F1_summary.md (지시서 §6.7, v4.2 F1~F3)

- 대상(F1): 주 분석 집합(S17, config `primary_subset`) 질의에서 기준 실행(기본 R8) 상위 10 에 들지 못한
  **등급 2 단위 전체** (v4.3 §6). 등급 2 단위 일부만 찾은
  질의의 나머지 단위도 포함한다. 놓친 단위마다 한 행, R2·R4·R6·R7·R8 순위(100위 밖이면 ">100").
- 분류 힌트(F2) `category_hint`: 순위 기록으로 판단할 수 있는 것만 자동으로 단다(쉼표 구분).
  최종 `category`·`note` 는 사용자가 채운다. 다시 실행하면 같은 (qid, unit_id) 의 기존 분류를 보존한다.
- 요약(F3): 최종 분류별 단위 수와 비율만 낸다. 원문·질의 문장은 넣지 않는다.
  `F1_failures.csv` 는 질의 문장을 담으므로 P0 대상이다(열지 않음).
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ev4.io import ExperimentConfig, read_qrels, read_queries, read_run
from ev4.stats import percentile
from src.index import case_index

CATEGORIES = ("토큰화(정확 일치 실패)", "교차 언어", "표현 불일치", "긴 단위 희석", "재순위 강등",
              "헤더 영향", "판정 이견", "기타")
# 순위 기록으로 자동 판단하는 힌트 (논문 [표 19] 중 원문이 필요 없는 것)
HINTS = ("재순위 강등", "헤더 영향", "교차 언어", "긴 단위 희석", "판정 이견")
DEFAULT_RANK_RUNS = ["R2", "R4", "R6", "R7", "R8"]
CUTOFF = 10
_NO_LANG = ("", "und")


def _read_existing(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {(r["qid"], r["unit_id"]): (r.get("category", ""), r.get("note", "")) for r in csv.DictReader(f)}


def unit_tokens_p95(cfg: ExperimentConfig, idx: Any) -> float:
    """T1 의 전체 단위 토큰 p95 — results/corpus.json 의 unit_tokens_p95_all, 없으면 색인에서 계산"""
    path = cfg.results_dir / "corpus.json"
    if path.exists():
        try:
            v = json.loads(path.read_text(encoding="utf-8")).get("unit_tokens_p95_all")
            if v is not None:
                return float(v)
        except (ValueError, OSError):
            pass
    sums = [sum(idx.parts[i].n_tokens for i in rows) for rows in idx.unit_rows.values()]
    return percentile(sums, 95)


def _within(rank: int | str, cutoff: int = CUTOFF) -> bool | None:
    """순위가 cutoff 이내인지 (실행이 없으면 None)"""
    if rank == "":
        return None
    return isinstance(rank, int) and rank <= cutoff


def category_hints(
    ranks: dict[str, int | str],
    query_lang: str,
    unit_lang: str,
    n_parts: int,
    n_tokens: int,
    p95: float,
    disputed: bool,
) -> list[str]:
    """분류 힌트 규칙 (v4.2 F2)

    - 재순위 강등: R6 ≤ 10 이고 R8 > 10
    - 헤더 영향: R7 ≤ 10 이고 R8 > 10
    - 교차 언어: query_lang ≠ 단위 lang 이고 R4 > 10 (단위 언어가 und·빈칸이면 판단하지 않음)
    - 긴 단위 희석: 파트가 2개 이상이거나 토큰 수가 T1 p95 초과
    - 판정 이견: A·B 등급이 달랐거나 중재로 등급이 바뀜
    """
    r8 = _within(ranks.get("R8", ""))
    out: list[str] = []
    if r8 is False and _within(ranks.get("R6", "")):
        out.append("재순위 강등")
    if r8 is False and _within(ranks.get("R7", "")):
        out.append("헤더 영향")
    if unit_lang not in _NO_LANG and query_lang != unit_lang and _within(ranks.get("R4", "")) is False:
        out.append("교차 언어")
    if n_parts >= 2 or n_tokens > p95:
        out.append("긴 단위 희석")
    if disputed:
        out.append("판정 이견")
    return out


def summary_markdown(base: str, rows: list[dict[str, Any]], n_failed_q: int) -> str:
    """F1 요약 — 최종 분류별 단위 수·비율 (+ 힌트 집계). 원문·질의 문장·실명 없음"""
    total = len(rows)
    cats = Counter(r["category"] for r in rows if r["category"])
    done = sum(cats.values())

    def _ratio(n: int, d: int) -> str:
        return f"{n / d:.1%}" if d else "–"

    md = [
        f"# F1 실패 분석 요약 — 기준 {base}, 상위 {CUTOFF} 밖 등급 2 단위", "",
        f"- 놓친 등급 2 단위 {total}건 (질의 {n_failed_q}건) · 분류 완료 {done}건 · 미분류 {total - done}건", "",
        "| 분류 | 단위 수 | 비율(분류 완료 기준) |", "|---|---:|---:|",
        *[f"| {c} | {cats.get(c, 0)} | {_ratio(cats.get(c, 0), done)} |" for c in CATEGORIES],
    ]
    unknown = {c: n for c, n in cats.items() if c not in CATEGORIES}
    md += [f"| {c} (목록 외) | {n} | {_ratio(n, done)} |" for c, n in sorted(unknown.items())]
    md += [f"| 계 | {done} | {_ratio(done, done)} |"]
    hints = Counter(h for r in rows for h in filter(None, r["category_hint"].split(",")))
    md += ["", "### 자동 분류 힌트 (참고, 한 단위에 여러 개 가능)", "",
           "| 힌트 | 단위 수 | 비율(놓친 단위 기준) |", "|---|---:|---:|",
           *[f"| {h} | {hints.get(h, 0)} | {_ratio(hints.get(h, 0), total)} |" for h in HINTS],
           f"| 힌트 없음 | {sum(1 for r in rows if not r['category_hint'])} | "
           f"{_ratio(sum(1 for r in rows if not r['category_hint']), total)} |"]
    md += ["", f"분류 선택지: {' / '.join(CATEGORIES)} — `F1_failures.csv` 의 category 칸에 입력"]
    return "\n".join(md) + "\n"


def build_failures(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """실패 분석 시트·요약 생성

    Returns:
        {n_queries_failed, n_rows, categories, hints}
    """
    from ev4.judgments import disputed_units

    base = str(cfg.extra.get("failure_run", "R8"))
    rank_runs = [r for r in cfg.extra.get("failure_rank_runs", DEFAULT_RANK_RUNS) if r in cfg.runs]
    if base not in rank_runs:
        rank_runs.append(base)
    cfg.check_subset_lock()
    queries = {q.qid: q for q in read_queries(cfg.queries_path, split="test")
               if q.support_group == cfg.primary_subset}
    qrels = read_qrels(cfg.qrels_path)
    runs = {r: read_run(cfg.run_path(r, "test")) for r in rank_runs if cfg.run_path(r, "test").exists()}
    if base not in runs:
        raise FileNotFoundError(f"기준 실행 {base} 파일이 없습니다: {cfg.run_path(base, 'test')}")
    idx = case_index.load(cfg.case_id)
    p95 = unit_tokens_p95(cfg, idx)
    disputed = disputed_units(cfg)
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
            n_tok = sum(p.n_tokens for p in parts)
            row: dict[str, Any] = {
                "qid": qid, "type": q.type, "query": q.text, "unit_id": uid,
                "unit_type": parts[0].unit_type if parts else "", "lang": meta.get("lang", ""),
                "n_parts": len(parts), "n_tokens": n_tok,
            }
            ranks: dict[str, int | str] = {}
            for r in rank_runs:
                if r not in runs:
                    ranks[r] = ""
                    continue
                ranked = runs[r].get(qid, [])
                ranks[r] = ranked.index(uid) + 1 if uid in ranked[: cfg.depth] else f">{cfg.depth}"
                row[f"rank_{r}"] = ranks[r]
            row["category_hint"] = ",".join(category_hints(
                ranks, q.query_lang, str(meta.get("lang", "")), len(parts), n_tok, p95, (qid, uid) in disputed))
            cat, note = prev.get((qid, uid), ("", ""))
            row["category"], row["note"] = cat, note
            rows.append(row)

    cols = ["qid", "type", "query", "unit_id", "unit_type", "lang", "n_parts", "n_tokens",
            *[f"rank_{r}" for r in rank_runs], "category_hint", "category", "note"]
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    (cfg.results_dir / "F1_summary.md").write_text(summary_markdown(base, rows, len(failed_q)), encoding="utf-8")
    cats = Counter(r["category"] for r in rows if r["category"])
    hints = Counter(h for r in rows for h in filter(None, r["category_hint"].split(",")))
    log(f"[failures] 놓친 등급 2 단위 {len(rows)}건 (질의 {len(failed_q)}) · 힌트 {dict(hints)} → {out}")
    return {"n_queries_failed": len(failed_q), "n_rows": len(rows), "categories": dict(cats),
            "hints": dict(hints), "p95_tokens": p95}
