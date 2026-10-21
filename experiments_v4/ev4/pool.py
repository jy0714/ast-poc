"""판정 시트 — pool_runs 상위 pool_depth 합집합 → pool_A.csv / pool_B.csv (지시서 §6.3)

- 블라인드: 시트에는 어느 실행이 찾았는지 넣지 않는다. 출처는 runs/pool_provenance.jsonl.
- 질의 안에서 단위 순서를 seed 고정으로 섞는다.
- A 는 전체 질의(45개), B 는 support_group 별 층화로 config `b_queries`(search 4·meta 1·nonsearch 6)만큼
  seed 고정으로 고른 질의의 전체 단위를 판정한다 (v4.3 §5).
- 판정자 C(v4.4 §9, config `c_queries` 가 있을 때만): B 와 겹치지 않는 질의를 같은 층화·다른 seed 로 골라
  pool_C.csv 를 만든다. 일치도는 A–B, A–C 를 따로 낸다.
- CSV 는 Excel 호환 UTF-8 BOM.
- 판정 기준 문서 data/judgments/판정기준.md 를 함께 만든다 (v4.2 J2).
- 풀 통계(질의당 대상 수, 번역 단위 수, 잘라 보인 단위 수) → results/J1_pool.md (v4.2 J1).
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ev4.io import (
    QUERY_TYPE_DEFS,
    QUERY_TYPE_NAMES,
    QUERY_TYPES,
    SUPPORT_GROUPS,
    ExperimentConfig,
    Query,
    read_run,
    write_jsonl,
)
from ev4.translate import NO_TRANSLATE_LANGS, Translator
from src.index import case_index

POOL_COLUMNS = ["qid", "query_text", "query_type", "unit_id", "unit_type", "date", "lang", "header",
                "text", "translation_ko", "grade", "annotator", "note"]
TRUNC_NOTE = "…(이하 생략, 전체: unit_id로 조회)"
CRITERIA_NAME = "판정기준.md"
# 논문 [표 9] 등급 기준 (v4.2 J2)
GRADE_CRITERIA = [
    (2, "결정적", "이 단위만으로 질의에 답할 수 있거나, 조사 결론을 직접 뒷받침한다"),
    (1, "관련", "질의와 관련되어 감사인이 검토해야 하지만, 단독으로는 결론 근거가 되지 않는다"),
    (0, "무관", "질의와 관련이 없다"),
]
# 집계·비교형 질문 판정 안내 (v4.3 §5, 사용자 확정)
AGGREGATE_GUIDE = ("질문이 여러 자료를 모아야 답할 수 있는 형태이더라도, 그 답을 단독으로 보여 주는 단위"
                   "(예: 입찰 참여 업체가 정리된 표, 견적 조건 변경을 지시한 메일)는 2, 답을 찾는 단서가 되는 단위는 1, "
                   "관련 없는 단위는 0으로 매긴다.")


def criteria_markdown() -> str:
    """판정 기준 문서 본문 (등급 기준 + 질의 3유형 정의)"""
    lines = ["# 판정 기준", "", "## 등급", "", "| 등급 | 구분 | 기준 |", "|---|---|---|"]
    lines += [f"| {g} | {name} | {desc} |" for g, name, desc in GRADE_CRITERIA]
    lines += ["", "등급은 원문 기준으로 매긴다.", "번역은 참고용이다.", "",
              "### 집계·비교를 요구하는 질문 [사용자 확정]", "", f"> {AGGREGATE_GUIDE}", "",
              "## 질의 유형 (search 집합 17개에만 적용, 나머지 질의는 `none`)", "",
              "| `type` | 이름 | 정의 | 주요 검색 신호 |", "|---|---|---|---|"]
    lines += [f"| `{t}` | {QUERY_TYPE_NAMES[t]} | {QUERY_TYPE_DEFS[t][0]} | {QUERY_TYPE_DEFS[t][1]} |"
              for t in QUERY_TYPES]
    return "\n".join(lines) + "\n"


def write_criteria(cfg: ExperimentConfig) -> Path:
    """판정 기준 문서를 판정 폴더에 쓴다"""
    path = cfg.judgments_dir / CRITERIA_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(criteria_markdown(), encoding="utf-8")
    return path


def pool_stats_markdown(stats: dict[str, Any]) -> str:
    """J1 풀 통계 표 (건수만, 질의 문장·본문 없음)"""
    lim = f"{stats['text_max_chars']:,}"
    lines = [
        "# J1 판정 풀 통계", "",
        f"- 풀 구성: 실행 {stats['n_runs']}개 × 상위 {stats['pool_depth']} 합집합 · 질의 {stats['n_queries']}건",
        "", "| 항목 | 값 |", "|---|---:|",
        f"| 질의당 판정 대상 단위 수 평균 | {stats['per_query_mean']:.1f} |",
        f"| 질의당 판정 대상 단위 수 최소 | {stats['per_query_min']} |",
        f"| 질의당 판정 대상 단위 수 최대 | {stats['per_query_max']} |",
        f"| 판정 대상 (질의, 단위) 행 수 | {stats['rows_A']} |",
        f"| 판정 대상 고유 단위 수 | {stats['unique_units']} |",
        f"| 번역을 함께 보인 행 수 | {stats['translated_rows']} |",
        f"| 번역을 함께 보인 고유 단위 수 | {stats['translated_units']} |",
        f"| {lim}자를 넘어 잘라 보인 행 수 | {stats['truncated_rows']} |",
        f"| {lim}자를 넘어 잘라 보인 고유 단위 수 | {stats['truncated_units']} |",
        f"| B 판정 질의 수 | {stats['queries_B']} |",
        f"| B 판정 행 수 | {stats['rows_B']} |",
        "",
        "B 판정 질의 (qid): " + (", ".join(stats.get("qids_B", [])) or "없음"),
    ]
    if stats.get("queries_C"):
        lines += ["", f"C 판정 질의 {stats['queries_C']}건 · 행 {stats['rows_C']} (B 와 겹치지 않음, v4.4 §9): "
                  + ", ".join(stats["qids_C"])]
    return "\n".join(lines) + "\n"


def select_b_queries(queries: list[Query], quota: dict[str, int], seed: int) -> list[str]:
    """support_group 별 층화 표집 (v4.3 §5) — 집합마다 quota 개 (집합 질의 수보다 많으면 전부), seed 고정"""
    by_group: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        by_group[q.support_group].append(q.qid)
    rng = random.Random(seed)
    chosen: list[str] = []
    for g in SUPPORT_GROUPS:
        qids = sorted(by_group.get(g, []))
        chosen += rng.sample(qids, min(len(qids), max(0, quota.get(g, 0))))
    return sorted(chosen)


def select_c_queries(queries: list[Query], quota: dict[str, int], seed: int, exclude: set[str]) -> list[str]:
    """판정자 C 질의 (v4.4 §9) — B 와 겹치지 않게, 같은 층화, 다른 seed(seed + 1)"""
    return select_b_queries([q for q in queries if q.qid not in exclude], quota, seed + 1)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POOL_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def build_pool(
    cfg: ExperimentConfig,
    queries: list[Query],
    split: str = "test",
    translate: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """판정 시트·판정 기준 문서·풀 통계 생성

    Returns:
        {rows_A, rows_B, queries_B, translated, per_query_mean/min/max, unique_units,
         translated_rows/units, truncated_rows/units, ...}
    """
    idx = case_index.load(cfg.case_id)
    runs = {name: read_run(cfg.run_path(name, split)) for name in cfg.pool_runs}
    translator = Translator(cfg.cache_dir / "translations.jsonl") if translate else None
    b_qids = set(select_b_queries(queries, cfg.b_queries, cfg.seed))
    c_qids = set(select_c_queries(queries, cfg.c_queries, cfg.seed, b_qids)) if cfg.c_queries else set()

    rows_a: list[dict] = []
    provenance: list[dict] = []
    per_query: list[int] = []
    truncated: set[tuple[str, str]] = set()
    for q in queries:
        found: dict[str, dict[str, int]] = defaultdict(dict)
        for name, run in runs.items():
            for rank, uid in enumerate(run.get(q.qid, [])[: cfg.pool_depth], 1):
                found[uid][name] = rank
        units = sorted(found)
        random.Random(f"{cfg.seed}:{q.qid}").shuffle(units)
        for uid in units:
            parts = idx.unit_parts(uid)
            if not parts:
                raise KeyError(f"색인에 없는 단위 {uid} (실행 파일과 색인이 다름 — 재실행 필요)")
            meta = parts[0].meta
            text = idx.unit_text(uid)
            if len(text) > cfg.text_max_chars:
                text = text[: cfg.text_max_chars] + TRUNC_NOTE
                truncated.add((q.qid, uid))
            lang = str(meta.get("lang", ""))
            ko = translator.translate(text) if translator and lang not in NO_TRANSLATE_LANGS else ""
            rows_a.append({
                "qid": q.qid, "query_text": q.text, "query_type": q.type,
                "unit_id": uid, "unit_type": parts[0].unit_type, "date": meta.get("date", ""),
                "lang": lang, "header": parts[0].header, "text": text, "translation_ko": ko,
                "grade": "", "annotator": "", "note": "",
            })
            provenance.append({"qid": q.qid, "unit_id": uid, "found_by": found[uid]})
        per_query.append(len(units))
        log(f"  {q.qid}: 판정 대상 {len(units)}건" + (" (B)" if q.qid in b_qids else ""))
    if translator:
        translator.save()
    rows_b = [r for r in rows_a if r["qid"] in b_qids]
    rows_c = [r for r in rows_a if r["qid"] in c_qids]
    _write_csv(cfg.judgments_dir / "pool_A.csv", rows_a)
    _write_csv(cfg.judgments_dir / "pool_B.csv", rows_b)
    if c_qids:
        _write_csv(cfg.judgments_dir / "pool_C.csv", rows_c)
    write_jsonl(cfg.runs_dir / "pool_provenance.jsonl", provenance)
    write_criteria(cfg)
    n_tr = translator.n_new if translator else 0
    tr_rows = [r for r in rows_a if r["translation_ko"]]
    stats: dict[str, Any] = {
        "rows_A": len(rows_a), "rows_B": len(rows_b), "queries_B": len(b_qids), "qids_B": sorted(b_qids),
        "rows_C": len(rows_c), "queries_C": len(c_qids), "qids_C": sorted(c_qids),
        "translated": n_tr,
        "n_runs": len(runs), "pool_depth": cfg.pool_depth, "n_queries": len(queries),
        "per_query_mean": sum(per_query) / len(per_query) if per_query else 0.0,
        "per_query_min": min(per_query, default=0), "per_query_max": max(per_query, default=0),
        "unique_units": len({r["unit_id"] for r in rows_a}),
        "translated_rows": len(tr_rows), "translated_units": len({r["unit_id"] for r in tr_rows}),
        "text_max_chars": cfg.text_max_chars,
        "truncated_rows": len(truncated), "truncated_units": len({u for _q, u in truncated}),
    }
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "J1_pool.md").write_text(pool_stats_markdown(stats), encoding="utf-8")
    log(f"[pool] A {len(rows_a)}행 · B {len(rows_b)}행 (질의 {len(b_qids)}건) · 새 번역 {n_tr}건 · "
        f"질의당 {stats['per_query_mean']:.1f}건 ({stats['per_query_min']}~{stats['per_query_max']}) "
        f"→ {cfg.judgments_dir}")
    log(f"[pool] B 판정 질의: {', '.join(sorted(b_qids))}")
    if c_qids:
        log(f"[pool] C 판정 질의: {', '.join(sorted(c_qids))} (C {len(rows_c)}행)")
    return stats
